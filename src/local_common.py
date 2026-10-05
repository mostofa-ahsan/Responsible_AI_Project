"""Shared helpers for the fully local evaluation layer (src/local_*.py, scripts/run_all_localeval.sh).

- forbid_api_clients(): hard guard. Requires LLM_OFFLINE=1 and makes constructing any Anthropic / OpenAI
  client raise, so no stage can reach an external API by accident.
- Systems: 3 base models + 3 x 1-epoch fine-tunes (run_2026-10-03) + 3 models x 3 epochs (run_epochs),
  answers restricted to the fixed ids in data/splits/eval_subset.json (500 / 500 / all 279 seen facts).
- run_worker(): runs src/vllm_json_worker.py under .venv-vllm (resumable, deadline-aware).
- Logging goes to logs/localeval.log with timestamps; stage markers live in results/local_eval/.done/.
"""

import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from utils import load_config, repo_path


def forbid_api_clients():
    assert os.environ.get("LLM_OFFLINE") == "1", "local evaluation must run with LLM_OFFLINE=1"

    def _blocked(self, *a, **k):
        raise AssertionError("API client construction is forbidden in the local evaluation (LLM_OFFLINE=1)")
    for mod, names in (("anthropic", ("Anthropic", "AsyncAnthropic", "AnthropicBedrock", "AnthropicVertex")),
                       ("openai", ("OpenAI", "AsyncOpenAI", "AzureOpenAI"))):
        try:
            m = __import__(mod)
        except Exception:
            continue
        for n in names:
            cls = getattr(m, n, None)
            if cls is not None:
                cls.__init__ = _blocked


forbid_api_clients()

CFG = load_config()
OUT = repo_path("results/local_eval")
DONE = OUT / ".done"
PER_ITEM = OUT / "per_item"
CACHE = OUT / "judge_cache"
KEYFACTS = repo_path("data/eval_keyfacts")
PACK = repo_path("results/paper_pack")
for d in (OUT, DONE, PER_ITEM, CACHE, KEYFACTS):
    d.mkdir(parents=True, exist_ok=True)

SPLITS = ["test_indomain", "test_heldout_docs", "test_seen_facts"]
FAMILIES = ["qwen3-8b", "gemma-4-e4b-it", "llama-3.1-8b-instruct"]
FAMILY_LABEL = {"qwen3-8b": "Qwen3-8B", "gemma-4-e4b-it": "Gemma 4 E4B", "llama-3.1-8b-instruct": "Llama 3.1 8B"}
# Okabe-Ito (colorblind-safe), one color per family
FAMILY_COLOR = {"qwen3-8b": "#0072B2", "gemma-4-e4b-it": "#D55E00", "llama-3.1-8b-instruct": "#009E73"}
VARIANTS = ["base", "ft1run", "ep1", "ep2", "ep3"]
VARIANT_LABEL = {"base": "base", "ft1run": "1-epoch (run 10-03)", "ep1": "ep1", "ep2": "ep2", "ep3": "ep3"}
OPUS_VARIANTS = {"base": "base", "ft1run": "finetuned"}

JUDGES = {
    "A": {"name": "mistral-small-3.2-24b-awq", "model": "jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym",
          "mistral3": True},
    "B": {"name": "phi-4-awq", "model": "stelterlab/phi-4-AWQ", "mistral3": False},
}
VLLM_PY = repo_path(".venv-vllm/bin/python")

_log = None


def log():
    global _log
    if _log is None:
        _log = logging.getLogger("localeval")
        _log.setLevel(logging.INFO)
        fh = logging.FileHandler(repo_path("logs/localeval.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        _log.addHandler(fh)
        if sys.stderr.isatty():     # under the orchestrator stderr already goes to logs/localeval.log
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
            _log.addHandler(sh)
    return _log


def systems():
    return [(f, v) for f in FAMILIES for v in VARIANTS]


def sys_id(fam, var):
    return f"{fam}__{var}"


def pred_path(fam, var, split):
    if var in OPUS_VARIANTS:
        return repo_path("results/run_2026-10-03") / fam / OPUS_VARIANTS[var] / f"{split}_predictions.jsonl"
    return repo_path("results/run_epochs") / fam / f"epoch{var[2:]}" / f"{split}_predictions.jsonl"


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


_items = None


def items():
    """{split: {qa_id: row}} for the eval-subset ids; seen-facts rows get evidence via source_qa_id."""
    global _items
    if _items is not None:
        return _items
    sub = json.loads(repo_path("data/splits/eval_subset.json").read_text())["splits"]
    full = None
    _items = {}
    for s in SPLITS:
        rows = {r["qa_id"]: r for r in read_jsonl(repo_path("data/splits") / f"{s}.jsonl")}
        ids = sub.get(s) or list(rows)
        if s == "test_seen_facts":
            if full is None:
                full = {r["qa_id"]: r for r in read_jsonl(repo_path("data/qa_pairs/full_v1.jsonl"))}
            for r in rows.values():
                src = full.get(r.get("source_qa_id"), {})
                r.setdefault("evidence", src.get("evidence", ""))
                r.setdefault("unit_ids", src.get("unit_ids", []))
        _items[s] = {q: rows[q] for q in ids}
    return _items


_preds = {}


def preds(fam, var, split):
    k = (fam, var, split)
    if k not in _preds:
        _preds[k] = {r["qa_id"]: r["prediction"] for r in read_jsonl(pred_path(fam, var, split))}
    return _preds[k]


def opus_grades(fam, var, split):
    if var not in OPUS_VARIANTS:
        return {}
    p = repo_path("results/run_2026-10-03") / fam / OPUS_VARIANTS[var] / f"{split}_grades.jsonl"
    return {r["qa_id"]: r for r in read_jsonl(p)}


def done(name):
    return (DONE / name).exists()


def mark(name, info=None):
    (DONE / name).write_text(json.dumps(info or {}, indent=2))


PAUSE = repo_path("results/.gpu_pause")       # created by scripts/gpu_watchdog.sh when the GPU runs hot


def pause_wait(tag="gpu"):
    """Thermal pause: while the watchdog's flag exists, sleep in 5-minute steps (called between chunks)."""
    while PAUSE.exists():
        log().info(f"[{tag}] GPU pause flag set (>= 84 C): sleeping 5 min")
        time.sleep(300)


def deadline_from_env(default_min):
    """DEADLINE (unix seconds) set by the orchestrator per stage; fallback: now + default."""
    return float(os.environ.get("STAGE_DEADLINE") or time.time() + default_min * 60)


def run_worker(model, rows, out_path, mode="json", schema=None, max_tokens=300, deadline=None, extra=None,
               max_model_len=4096, gpu_util=0.90, label="worker"):
    """rows: [{"id", "messages"}]. Results are appended to out_path ({"id", "text"} or {"id", "p_yes"}).
    Only ids missing from out_path are sent. Returns the number still missing afterwards."""
    have = {r["id"] for r in read_jsonl(out_path)}
    todo = [r for r in rows if r["id"] not in have]
    if not todo:
        return 0
    inp = Path(str(out_path) + ".in.jsonl")
    write_jsonl(inp, todo)
    cmd = [str(VLLM_PY), str(repo_path("src/vllm_json_worker.py")), "--model", model, "--input", str(inp),
           "--output", str(out_path), "--mode", mode, "--max-tokens", str(max_tokens),
           "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_util)]
    if schema is not None:
        sp = Path(str(out_path) + ".schema.json")
        sp.write_text(json.dumps(schema))
        cmd += ["--schema", str(sp)]
    if deadline:
        cmd += ["--deadline", str(deadline)]
    cmd += extra or []
    log().info(f"[{label}] vLLM worker: {len(todo)} prompts -> {Path(out_path).name}")
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0", LLM_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=str(repo_path("src/vllm_shims")))
    with repo_path("logs/localeval_worker.log").open("a") as lf:
        lf.write(f"\n===== {time.strftime('%F %T')} {label}: {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=lf).returncode
    have = {r["id"] for r in read_jsonl(out_path)}
    missing = sum(1 for r in rows if r["id"] not in have)
    log().info(f"[{label}] worker exit {rc}; {len(rows) - missing}/{len(rows)} done")
    inp.unlink(missing_ok=True)
    return missing


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except Exception:
                return None
    return None


def words(t):
    return len((t or "").split())


def shuffled(rows, seed=0):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows
