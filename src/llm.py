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
