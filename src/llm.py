"""LLM client wrapper: one complete() for all pipeline stages.

    from llm import LLM
    llm = LLM(cfg, stage="qa_extract")
    res = llm.complete(prompt, system, role="generator", json_schema=MyPydanticModel)
    res.parsed / res.text / res.cost / res.cached / res.served_model

Providers (config models.<role>.provider):
  - anthropic: Messages API via the official SDK; structured output with
    messages.parse(output_format=<pydantic model>); optional server-side refusal
    fallback (fallbacks: default).
  - openai_compatible: any /v1/chat/completions endpoint (vLLM, Ollama, ...);
    JSON mode via response_format json_schema, validated with pydantic.

Features: SDK retries with exponential backoff (429/5xx/connection), a simple
request-start rate limit, one re-ask on schema-validation failure, token and
cost tracking per call (appended to llm.usage_log), and a disk cache keyed on
(provider, model, settings, system, prompt, schema) so re-runs don't re-pay.
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from utils import repo_path

load_dotenv(repo_path(".env"))


@dataclass
class Result:
    text: str
    parsed: BaseModel | None
    model: str
    served_model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    cached: bool = False
    stop_reason: str = ""
    extra: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


# --- Message Batches -----------------------------------------------------------
# Collect mode: with LLM_COLLECT=<path>, complete() appends every uncached request to that JSONL
# file and raises Deferred instead of calling the API. The orchestrator (src/full_run.py) submits
# the collected requests as one batch, writes each succeeded result into the same disk cache
# (same key as complete() would use) and llm_usage.jsonl at batch prices, then re-runs the stage;
# cached calls then return immediately, so a stage finishes after one batch per dependency round.



class Deferred(Exception):
    """Raised in collect mode for an uncached request (recorded for a batch)."""


class Unavailable(Exception):
    """Raised with LLM_OFFLINE=1 for an uncached request (budget cap reached: no more API calls)."""


class BillingError(LLMError):
    """Credit balance or spend limit reached: stop the run cleanly."""


BILLING_RX = re.compile(r"credit balance|billing|spend(ing)? limit|usage limit|insufficient (credit|fund)|"
                         r"payment required|quota", re.I)


def is_billing_error(text):
    return bool(BILLING_RX.search(str(text or "")))


def _schema_ref(schema):
    mod = schema.__module__
    if mod == "__main__":
        mod = Path(sys.modules["__main__"].__file__).stem
    return f"{mod}:{schema.__qualname__}"



class LLM:
    def __init__(self, cfg, stage):
        self.cfg, self.stage = cfg, stage
        lcfg = cfg["llm"]
        self.cache_dir = repo_path(lcfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_log = repo_path(lcfg["usage_log"])
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        self.min_interval = lcfg["min_interval_s"]
        self.max_retries = lcfg["max_retries"]
        self.validation_retries = lcfg["validation_retries"]
        self._lock = threading.Lock()
        self._last_start = 0.0
        self._clients = {}
        self.totals = {}   # model -> {calls, cached, input, output, cost}

    # --- public ---------------------------------------------------------
    def complete(self, prompt, system="", role="generator", json_schema=None, model=None):
        """Cached completion. In collect mode (env LLM_COLLECT=<jsonl>), an uncached request is
        recorded for a Message Batch and Deferred is raised instead of calling the API."""
        collect = os.environ.get("LLM_COLLECT")
        if collect:
            mcfg = dict(self.cfg["models"][role])
            if model:
                mcfg["model"] = model
            key = self._cache_key(mcfg, system, prompt, json_schema)
            if not (self.cache_dir / f"{key}.json").exists():
                rec = {"cache_key": key, "stage": self.stage, "role": role, "mcfg": mcfg, "system": system,
                       "prompt": prompt, "schema": _schema_ref(json_schema) if json_schema else None}
                with self._lock, open(collect, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raise Deferred(key)
        if os.environ.get("LLM_OFFLINE"):
            mcfg = dict(self.cfg["models"][role])
            if model:
                mcfg["model"] = model
            key = self._cache_key(mcfg, system, prompt, json_schema)
            if not (self.cache_dir / f"{key}.json").exists():
                raise Unavailable(key)
        try:
            return self._complete_standard(prompt, system, role, json_schema, model)
        except LLMError:
            raise
        except Exception as e:
            if is_billing_error(e):
                raise BillingError(str(e)) from e
            raise

    def _complete_standard(self, prompt, system="", role="generator", json_schema=None, model=None):
        mcfg = dict(self.cfg["models"][role])
        if model:
            mcfg["model"] = model
        key = self._cache_key(mcfg, system, prompt, json_schema)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            res = Result(**{k: v for k, v in data.items() if k != "parsed"}, parsed=None)
            res.parsed = json_schema.model_validate_json(res.text) if json_schema else None
            res.cached = True
            self._record(res, role)
            return res

        last_err = None
        for attempt in range(self.validation_retries + 1):
            p = prompt if attempt == 0 else (
                f"{prompt}\n\nYour previous reply did not match the required JSON schema "
                f"({last_err}). Reply again with valid JSON only.")
            res = self._call(mcfg, system, p, json_schema)
            try:
                if json_schema and res.parsed is None:
                    res.parsed = json_schema.model_validate_json(res.text)
                break
            except ValidationError as e:
                last_err = str(e).splitlines()[0]
                self._record(res, role)   # failed attempts still cost money
        else:
            raise LLMError(f"schema validation failed after retries: {last_err}")

        payload = {k: v for k, v in res.__dict__.items() if k not in ("parsed", "cached")}
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(cache_file)
        self._record(res, role)
        return res

    def summary(self):
        lines, total = [], 0.0
        for model, t in sorted(self.totals.items()):
            total += t["cost"]
            lines.append(f"  {model:28s} calls={t['calls']:4d} (cached {t['cached']:4d})  "
                         f"in={t['input']:>9,}  out={t['output']:>9,}  ${t['cost']:.4f}")
        lines.append(f"  {'TOTAL (this run, new calls)':28s} ${total:.4f}")
        return lines

    # --- internals ------------------------------------------------------
    def _cache_key(self, mcfg, system, prompt, schema):
        settings = {k: mcfg.get(k) for k in ("provider", "model", "effort", "max_tokens", "base_url")}
        blob = json.dumps({"s": settings, "sys": system, "p": prompt,
                           "schema": schema.model_json_schema() if schema else None}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _throttle(self):
        with self._lock:
            wait = self._last_start + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_start = time.time()

    def _call(self, mcfg, system, prompt, schema):
        self._throttle()
        if mcfg["provider"] == "anthropic":
            return self._call_anthropic(mcfg, system, prompt, schema)
        if mcfg["provider"] == "openai_compatible":
            return self._call_openai_compatible(mcfg, system, prompt, schema)
        raise LLMError(f"unknown provider {mcfg['provider']}")

    def _anthropic(self):
        if "anthropic" not in self._clients:
            import anthropic
            self._clients["anthropic"] = anthropic.Anthropic(max_retries=self.max_retries)
        return self._clients["anthropic"]

    def _call_anthropic(self, mcfg, system, prompt, schema):
        client = self._anthropic()
        kwargs = dict(model=mcfg["model"], max_tokens=mcfg.get("max_tokens", 16000),
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        if mcfg.get("effort"):
            kwargs["output_config"] = {"effort": mcfg["effort"]}
        if schema is not None:
            kwargs["output_format"] = schema
        if mcfg.get("fallbacks"):
            resp = client.beta.messages.parse(
                betas=["server-side-fallback-2026-07-01"], fallbacks=mcfg["fallbacks"], **kwargs)
        elif schema is not None:
            resp = client.messages.parse(**kwargs)
        else:
            resp = client.messages.create(**kwargs)

        if resp.stop_reason == "refusal":
            cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            raise LLMError(f"refusal (category={cat}) from {resp.model}")
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"hit max_tokens={kwargs['max_tokens']} on {resp.model}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        res = Result(
            text=text, parsed=getattr(resp, "parsed_output", None), model=mcfg["model"],
            served_model=resp.model, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            stop_reason=resp.stop_reason or "")
        res.cost = self._price(res)
        return res

    def _call_openai_compatible(self, mcfg, system, prompt, schema):
        import httpx2
        key = os.environ.get(mcfg.get("api_key_env", "OPENAI_COMPATIBLE_API_KEY"), "none")
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = {"model": mcfg["model"], "messages": msgs, "max_tokens": mcfg.get("max_tokens", 4096)}
        body.update(mcfg.get("extra_body", {}))   # e.g. chat_template_kwargs: {enable_thinking: false}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema()}}
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx2.post(f"{mcfg['base_url'].rstrip('/')}/chat/completions", json=body,
                                headers={"Authorization": f"Bearer {key}"}, timeout=600)
                if r.status_code in (408, 409, 429) or r.status_code >= 500:
                    raise LLMError(f"HTTP {r.status_code}")
                r.raise_for_status()
                break
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        data = r.json()
        u = data.get("usage", {})
        res = Result(text=data["choices"][0]["message"]["content"] or "", parsed=None,
                     model=mcfg["model"], served_model=data.get("model", mcfg["model"]),
                     input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                     stop_reason=data["choices"][0].get("finish_reason", ""))
        res.cost = self._price(res)
        return res

    def _price(self, res):
        p = self.cfg.get("pricing", {}).get(res.served_model) or self.cfg.get("pricing", {}).get(res.model)
        if not p:
            return 0.0
        return (res.input_tokens * p["input"] + res.output_tokens * p["output"]
                + res.cache_read_tokens * p.get("cache_read", 0)
                + res.cache_write_tokens * p.get("cache_write", 0)) / 1e6

    def _record(self, res, role):
        with self._lock:
            t = self.totals.setdefault(res.served_model or res.model,
                                       {"calls": 0, "cached": 0, "input": 0, "output": 0, "cost": 0.0})
            t["calls"] += 1
            if res.cached:
                t["cached"] += 1
            else:
                t["input"] += res.input_tokens
                t["output"] += res.output_tokens
                t["cost"] += res.cost
            with self.usage_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "stage": self.stage, "role": role,
                                    "model": res.model, "served_model": res.served_model,
                                    "input": res.input_tokens, "output": res.output_tokens,
                                    "cost": 0.0 if res.cached else round(res.cost, 6),
                                    "cached": res.cached}) + "\n")


def _load_schema(ref):
    import importlib
    mod, name = ref.split(":")
    return getattr(importlib.import_module(mod), name)


def _batch_params(rec):
    from anthropic.lib._parse._transform import transform_schema
    from pydantic import TypeAdapter
    m = rec["mcfg"]
    if m["provider"] != "anthropic":
        raise LLMError(f"batches need the anthropic provider, got {m['provider']}")
    params = {"model": m["model"], "max_tokens": m.get("max_tokens", 16000),
              "messages": [{"role": "user", "content": rec["prompt"]}]}
    if rec["system"]:
        params["system"] = rec["system"]
    oc = {}
    if m.get("effort"):
        oc["effort"] = m["effort"]
    if rec["schema"]:
        schema = TypeAdapter(_load_schema(rec["schema"])).json_schema()
        oc["format"] = {"type": "json_schema", "schema": transform_schema(schema)}
    if oc:
        params["output_config"] = oc
    return params            # fallbacks are not supported on batches; refusals are retried via standard


def custom_id_for(rec):
    h = hashlib.sha1(f"{rec['stage']}|{rec['cache_key']}".encode()).hexdigest()[:40]
    prefix = re.sub(r"[^a-zA-Z0-9]", "", rec["stage"].split(":")[0])[:16] or "req"
    return f"{prefix}-{h}"                       # matches ^[a-zA-Z0-9_-]{1,64}$


class BatchRunner:
    """Submit collected requests as one Message Batch, poll, and write results into the cache."""

    def __init__(self, cfg, log, status_cb=None):
        self.cfg, self.log = cfg, log
        self.dir = repo_path(cfg["llm"].get("batch_dir", "data/cache/batches"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status_cb = status_cb or (lambda **kw: None)
        self.discount = cfg["llm"].get("batch_discount", 0.5)
        self._client_obj = None

    def _client(self):
        # One long-lived client: a temporary client can be garbage-collected (closing its
        # connection) while a results stream from it is still being read.
        if self._client_obj is None:
            import anthropic
            self._client_obj = anthropic.Anthropic(max_retries=self.cfg["llm"]["max_retries"])
        return self._client_obj

    def pending(self, stage_prefix):
        """Submitted batches for this stage whose results were never collected (crash/restart)."""
        out = []
        for f in sorted(self.dir.glob("msgbatch_*.json")):
            meta = json.loads(f.read_text(encoding="utf-8"))
            if not meta.get("collected") and meta.get("stage_prefix") == stage_prefix:
                out.append(f.stem)
        return out

    def submit_batch(self, records, stage_prefix=""):
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        mapping, reqs = {}, []
        for rec in records:
            cid = custom_id_for(rec)
            if cid in mapping:
                continue
            mapping[cid] = rec
            reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**_batch_params(rec))))
        try:
            batch = self._client().messages.batches.create(requests=reqs)
        except Exception as e:
            if is_billing_error(e):
                raise BillingError(str(e)) from e
            raise
        (self.dir / f"{batch.id}.json").write_text(json.dumps(
            {"created": time.time(), "n": len(reqs), "stage_prefix": stage_prefix, "collected": False,
             "mapping": mapping}), encoding="utf-8")
        self.log.info(f"submitted batch {batch.id} with {len(reqs)} requests")
        return batch.id

    def poll(self, batch_id, interval_s=60, timeout_min=90):
        """Wait until the batch ends; cancel it after timeout_min. Returns (batch, timed_out)."""
        client = self._client()
        start, timed_out = time.time(), False
        errors = 0
        while True:
            try:
                b = client.messages.batches.retrieve(batch_id)
                errors = 0
            except Exception as e:  # noqa: BLE001 - transient network errors: keep polling
                if is_billing_error(e):
                    raise BillingError(str(e)) from e
                errors += 1
                self.log.warning(f"poll error ({errors}): {type(e).__name__}: {e}")
                if errors >= 10:
                    raise
                time.sleep(interval_s)
                continue
            rc = b.request_counts
            counts = {"processing": rc.processing, "succeeded": rc.succeeded, "errored": rc.errored,
                      "canceled": rc.canceled, "expired": rc.expired}
            elapsed = (time.time() - start) / 60
            self.log.info(f"batch {batch_id}: {b.processing_status} {counts} ({elapsed:.0f} min)")
            self.status_cb(batch_id=batch_id, batch_status=b.processing_status, request_counts=counts,
                           batch_elapsed_min=round(elapsed, 1))
            if b.processing_status == "ended":
                return b, timed_out
            if not timed_out and elapsed >= timeout_min:
                self.log.warning(f"batch {batch_id} not ended after {timeout_min} min: canceling")
                client.messages.batches.cancel(batch_id)
                timed_out = True
            time.sleep(interval_s if not timed_out else 15)

    def collect(self, batch_id, attempts=3):
        """collect_once() with retries; safe to repeat (already-cached results are skipped)."""
        for i in range(attempts):
            try:
                res = self.collect_once(batch_id)
                mp = self.dir / f"{batch_id}.json"
                meta = json.loads(mp.read_text(encoding="utf-8"))
                meta["collected"] = True
                mp.write_text(json.dumps(meta), encoding="utf-8")
                return res
            except BillingError:
                raise
            except Exception as e:  # noqa: BLE001
                self.log.warning(f"collect {batch_id} attempt {i + 1} failed: {type(e).__name__}: {e}")
                self._client_obj = None
                if i == attempts - 1:
                    raise
                time.sleep(10)

    def collect_once(self, batch_id):
        """Write succeeded results to cache + usage log. Returns (n_ok, failed_records, billing)."""
        meta = json.loads((self.dir / f"{batch_id}.json").read_text(encoding="utf-8"))
        mapping = meta["mapping"]
        usage_log = repo_path(self.cfg["llm"]["usage_log"])
        cache_dir = repo_path(self.cfg["llm"]["cache_dir"])
        failed, ok, billing, seen = [], 0, False, set()
        pricing = self.cfg.get("pricing", {})
        client = self._client()
        results = client.messages.batches.results(batch_id)
        with usage_log.open("a", encoding="utf-8") as ulog:
            for res in results:
                rec = mapping.get(res.custom_id)
                if rec is None:
                    continue
                seen.add(res.custom_id)
                if (cache_dir / f"{rec['cache_key']}.json").exists():
                    ok += 1               # written by an earlier (interrupted) collect: don't double-log
                    continue
                r = res.result
                if r.type != "succeeded":
                    err = getattr(getattr(r, "error", None), "error", None)
                    msg = f"{r.type}: {getattr(err, 'type', '')} {getattr(err, 'message', '')}"
                    billing |= is_billing_error(msg)
                    failed.append({**rec, "_why": msg})
                    continue
                msg = r.message
                if msg.stop_reason in ("refusal", "max_tokens"):
                    failed.append({**rec, "_why": f"stop_reason={msg.stop_reason}"})
                    continue
                text = "".join(b.text for b in msg.content if b.type == "text")
                if rec["schema"]:
                    try:
                        _load_schema(rec["schema"]).model_validate_json(text)
                    except ValidationError as e:
                        failed.append({**rec, "_why": f"schema: {str(e).splitlines()[0]}"})
                        continue
                u = msg.usage
                p = pricing.get(msg.model) or pricing.get(rec["mcfg"]["model"]) or {}
                cr = getattr(u, "cache_read_input_tokens", 0) or 0
                cw = getattr(u, "cache_creation_input_tokens", 0) or 0
                cost = self.discount * (u.input_tokens * p.get("input", 0) + u.output_tokens * p.get("output", 0)
                                        + cr * p.get("cache_read", 0) + cw * p.get("cache_write", 0)) / 1e6
                payload = {"text": text, "model": rec["mcfg"]["model"], "served_model": msg.model,
                           "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                           "cache_read_tokens": cr, "cache_write_tokens": cw, "cost": cost,
                           "stop_reason": msg.stop_reason or "", "extra": {"batch_id": batch_id}}
                f = cache_dir / f"{rec['cache_key']}.json"
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                tmp.replace(f)
                ulog.write(json.dumps({"ts": time.time(), "stage": rec["stage"], "role": rec["role"],
                                       "model": rec["mcfg"]["model"], "served_model": msg.model,
                                       "input": u.input_tokens, "output": u.output_tokens,
                                       "cost": round(cost, 6), "cached": False, "batch": True}) + "\n")
                ok += 1
        for cid, rec in mapping.items():          # anything without a result line (should not happen)
            if cid not in seen:
                failed.append({**rec, "_why": "no result"})
        return ok, failed, billing

    def retry_standard(self, failed, workers=8):
        """Send failed/unfinished requests once through the standard API (fallbacks enabled there)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        llms = {}

        def one(rec):
            llm = llms.setdefault(rec["stage"], LLM(self.cfg, rec["stage"]))
            schema = _load_schema(rec["schema"]) if rec["schema"] else None
            prev = os.environ.pop("LLM_COLLECT", None)
            try:
                return llm.complete(rec["prompt"], rec["system"], role=rec["role"], json_schema=schema,
                                    model=rec["mcfg"]["model"])
            finally:
                if prev:
                    os.environ["LLM_COLLECT"] = prev

        ok, still = 0, []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, rec): rec for rec in failed}
            for fut in as_completed(futs):
                try:
                    fut.result()
                    ok += 1
                except BillingError:
                    raise
                except Exception as e:  # noqa: BLE001 - reported to caller
                    still.append({**futs[fut], "_why": f"standard retry: {type(e).__name__}: {e}"})
        return ok, still


def batched_map(cfg, log, stage, fn, items, workers=8):
    """Run fn(item) for every item with its uncached LLM calls sent as ONE Message Batch.

    fn must make its LLM calls through LLM(cfg, <stage starting with `stage`>).complete(). Pass 1
    runs fn in collect mode (calls are recorded, items deferred); the recorded requests go out as a
    batch and are written into the cache; pass 2 runs fn normally (all cache hits) and returns
    {index: result or Exception}. Only one dependency level: fn's calls must not depend on each
    other's results (use full_run.py's rounds for multi-step stages)."""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    def run_all():
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {i: ex.submit(fn, it) for i, it in enumerate(items)}
        out = {}
        for i, f in futs.items():
            try:
                out[i] = f.result()
            except Exception as e:  # noqa: BLE001 - returned per item
                out[i] = e
        return out

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        collect = tmp.name
    prev = os.environ.get("LLM_COLLECT")
    os.environ["LLM_COLLECT"] = collect
    try:
        run_all()
    finally:
        if prev is None:
            os.environ.pop("LLM_COLLECT", None)
        else:
            os.environ["LLM_COLLECT"] = prev
    recs = {}
    with open(collect, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs.setdefault(r["cache_key"], r)
    os.unlink(collect)
    if recs:
        lcfg = cfg["llm"]
        runner = BatchRunner(cfg, log)
        for bid in runner.pending(stage):         # resume an interrupted earlier call
            runner.poll(bid, interval_s=lcfg["batch_poll_s"], timeout_min=lcfg["batch_timeout_min"])
            runner.collect(bid)
        todo = [r for k, r in recs.items() if not (repo_path(lcfg["cache_dir"]) / f"{k}.json").exists()]
        if todo:
            bid = runner.submit_batch(todo, stage_prefix=stage)
            runner.poll(bid, interval_s=lcfg["batch_poll_s"], timeout_min=lcfg["batch_timeout_min"])
            ok, failed, billing = runner.collect(bid)
            log.info(f"{stage}: batch {bid}: {ok} succeeded, {len(failed)} to retry via standard API")
            if billing:
                raise BillingError(f"billing error in batch {bid}")
            if failed:
                runner.retry_standard(failed, workers=lcfg["standard_workers"])
    return run_all()


DEFAULT_CALL_COST = {"claude-opus-5-5": 0.0065, "claude-sonnet-5-5": 0.003, "claude-haiku-4-5-20251001": 0.001}


def estimate_cost(cfg, records, batch=True):
    """Estimated USD for records: mean logged cost per call for the same stage+role (batch or not),
    else the same model, else DEFAULT_CALL_COST."""
    by_sr, by_model = {}, {}
    path = repo_path(cfg["llm"]["usage_log"])
    if path.exists():
        for line in path.open(encoding="utf-8"):
            u = json.loads(line)
            if u["cached"] or bool(u.get("batch")) != batch:
                continue
            by_sr.setdefault((u["stage"], u["role"]), []).append(u["cost"])
            by_model.setdefault(u["model"], []).append(u["cost"])
    total = 0.0
    for r in records:
        c = by_sr.get((r["stage"], r["role"])) or by_model.get(r["mcfg"]["model"])
        total += sum(c) / len(c) if c else DEFAULT_CALL_COST.get(r["mcfg"]["model"], 0.005)
    return total

