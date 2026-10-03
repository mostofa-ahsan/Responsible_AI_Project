"""Multi-model fine-tuning and closed-book evaluation run (driven by scripts/model_runs.sh).

Subcommands (each idempotent; scripts/model_runs.sh adds per-step done-markers):
  subset                 fixed stratified eval subset (q_type x dimension) -> data/splits/eval_subset.json
  download  --model M    disk check (>= MIN_FREE_GB free after download), then snapshot_download
  smoke     --model M    30-step QLoRA smoke test -> models/<name>/smoke_summary.json
  epochs                 one epoch count for all models from the smoke tests (2, or 1 if 3 models x 2
                         epochs > EPOCH_BUDGET_H hours) -> <run dir>/epochs.json
  train     --model M    QLoRA with the shared recipe -> models/<name>/adapter + train logs
  generate  --model M --arm base|finetuned --split S
                         greedy answers to every test question (vLLM, LoRA for finetuned; HF fallback)
  grade                  Opus judge (batch path) on the fixed subset for every predictions file;
                         a 50-answer pilot measures the cost per answer; the subset is shrunk uniformly
                         (same questions for every model and arm) if the eval budget would be exceeded
  report                 summaries, comparison.md/.csv, plot, environment.txt
Decisions go to logs/UNATTENDED_DECISIONS.md.
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from qa_common import read_jsonl
from utils import get_logger, load_config, repo_path

RUN = "run_2026-10-03"
MODELS = ["Qwen/Qwen3-8B", "google/gemma-4-E4B-it", "meta-llama/Llama-3.1-8B-Instruct"]
SPLITS = ["test_indomain", "test_heldout_docs"]
ARMS = ["base", "finetuned"]
SUBSET_PER_SPLIT = 500
MIN_FREE_GB = 10
EPOCH_BUDGET_H = 4.5
SEED = 42


def short(model_id):
    return model_id.split("/")[-1].lower()


def run_dir(cfg):
    d = repo_path("results") / RUN
    d.mkdir(parents=True, exist_ok=True)
    return d


def decision(cfg, step, msg, why):
    path = repo_path(cfg["paths"]["logs"]) / "UNATTENDED_DECISIONS.md"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"| {time.strftime('%H:%M')} | {step} | {msg} | {why} |\n")


def py(*args):
    return subprocess.run([sys.executable, *args], cwd=repo_path(".")).returncode


# --- subset ---------------------------------------------------------------------------------------
def build_subset(cfg, log):
    path = repo_path(cfg["paths"]["splits"]) / "eval_subset.json"
    if path.exists():
        log.info(f"eval subset exists: {path}")
        return json.loads(path.read_text())
    rng = random.Random(SEED)
    out = {"seed": SEED, "strata": "q_type x dimension", "per_split": SUBSET_PER_SPLIT, "splits": {}}
    for split in SPLITS:
        rows = read_jsonl(repo_path(cfg["paths"]["splits"]) / f"{split}.jsonl")
        strata = defaultdict(list)
        for r in rows:
            strata[(r["q_type"], r["dimension"])].append(r["qa_id"])
        for v in strata.values():
            rng.shuffle(v)
        # proportional quotas (largest remainder), then an interleaved order so any prefix is stratified
        n = min(SUBSET_PER_SPLIT, len(rows))
        raw = {k: n * len(v) / len(rows) for k, v in strata.items()}
        quota = {k: int(x) for k, x in raw.items()}
        for k in sorted(raw, key=lambda k: raw[k] - quota[k], reverse=True)[:n - sum(quota.values())]:
            quota[k] += 1
        picked = {k: strata[k][:quota[k]] for k in strata}
        order, i = [], 0
        while len(order) < n:
            for k in sorted(picked):
                if i < len(picked[k]):
                    order.append(picked[k][i])
            i += 1
        out["splits"][split] = order[:n]
        log.info(f"eval subset {split}: {len(order[:n])} of {len(rows)} questions, {len(strata)} strata")
    path.write_text(json.dumps(out, indent=2))
    return out


# --- download -------------------------------------------------------------------------------------
def download(cfg, model, log):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(model, files_metadata=True)
    need = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9
    try:   # already cached?
        p = snapshot_download(model, local_files_only=True,
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt", "tokenizer*"])
        if any(Path(p).glob("*.safetensors")):
            log.info(f"{model}: already cached at {p}")
            return 0
    except Exception:  # noqa: BLE001 - not cached yet
        pass
    free = shutil.disk_usage(repo_path(".")).free / 1e9
    if free - need < MIN_FREE_GB:
        decision(cfg, f"download {model}", f"skipped: {free:.1f} GB free, needs {need:.1f} GB + {MIN_FREE_GB} GB reserve",
                 "disk reserve")
        return 2
    try:
        snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt", "tokenizer*"])
    except Exception as e:  # noqa: BLE001 - gated / network
        decision(cfg, f"download {model}", f"failed: {type(e).__name__}: {str(e)[:120]}", "continue with the other models")
        return 1
    return 0


# --- epochs ---------------------------------------------------------------------------------------
def choose_epochs(cfg, log):
    smokes = {}
    for m in MODELS:
        p = repo_path("models") / short(m) / "smoke_summary.json"
        if p.exists():
            smokes[m] = json.loads(p.read_text())
    if not smokes:
        raise SystemExit("no smoke tests available")
    h1 = {m: s["projected_hours_per_epoch"] for m, s in smokes.items()}
    total2 = 2 * sum(h1.values())
    epochs = 2 if total2 <= EPOCH_BUDGET_H else 1
    msg = (f"epochs = {epochs} for all models (projected 2-epoch total {total2:.2f} h for {len(smokes)} models: "
           + ", ".join(f"{short(m)} {h:.2f} h/epoch" for m, h in h1.items()) + ")")
    decision(cfg, "epochs", msg, f"rule: 2 epochs unless the 2-epoch total exceeds {EPOCH_BUDGET_H} h")
    (run_dir(cfg) / "epochs.json").write_text(json.dumps({"epochs": epochs, "hours_per_epoch": h1,
                                                          "total_2_epoch_hours": round(total2, 2)}, indent=2))
    log.info(msg)
    return epochs


# --- generate -------------------------------------------------------------------------------------
def generate(cfg, model, arm, split, log):
    from eval_closedbook import HFGenerator, generate_vllm, messages_for, vllm_check
    out = run_dir(cfg) / short(model) / arm
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / f"{split}_predictions.jsonl"
    rows = read_jsonl(repo_path(cfg["paths"]["splits"]) / f"{split}.jsonl")
    done = {p["qa_id"] for p in read_jsonl(pred_path)} if pred_path.exists() else set()
    todo = [r for r in rows if r["qa_id"] not in done]
    if not todo:
        log.info(f"{model} {arm} {split}: all {len(rows)} predictions present")
        return 0
    adapter = None
    if arm == "finetuned":
        adapter = repo_path("models") / short(model) / "adapter"
        if not (adapter / "adapter_config.json").exists():
            log.error(f"{model}: no adapter at {adapter}")
            return 1
        adapter = str(adapter)
    system = cfg["train"]["system_prompt"]
    backend, t0 = "vllm", time.time()
    ok, why = vllm_check()
    preds = None
    if ok:
        try:
            preds = generate_vllm(cfg, todo, system, adapter, log, model=model)
        except Exception as e:  # noqa: BLE001 - fall back to HF
            decision(cfg, f"generate {short(model)} {arm} {split}", f"vLLM failed ({type(e).__name__}); used HF generate",
                     "fallback")
    else:
        decision(cfg, f"generate {short(model)} {arm} {split}", f"vLLM unavailable ({why}); used HF generate", "fallback")
    if preds is None:
        backend = "hf"
        gen = HFGenerator(cfg, adapter, False, model=model)
        preds = {}
        for i in range(0, len(todo), 16):
            batch = todo[i:i + 16]
            for r, o in zip(batch, gen.generate([messages_for(system, r["question"]) for r in batch],
                                                cfg["eval"]["max_new_tokens"])):
                preds[r["qa_id"]] = o
            log.info(f"HF generate {short(model)} {arm} {split}: {min(i + 16, len(todo))}/{len(todo)}")
    with pred_path.open("a", encoding="utf-8") as f:
        for r in todo:
            if r["qa_id"] in preds:
                f.write(json.dumps({"qa_id": r["qa_id"], "prediction": preds[r["qa_id"]]}, ensure_ascii=False) + "\n")
    meta = {"model": model, "arm": arm, "split": split, "backend": backend, "adapter": adapter,
            "n": len(rows), "seconds": round(time.time() - t0), "decoding": "greedy",
            "max_new_tokens": cfg["eval"]["max_new_tokens"], "thinking": "disabled"}
    (out / f"{split}_generation.json").write_text(json.dumps(meta, indent=2))
    log.info(f"{model} {arm} {split}: {len(preds)} predictions via {backend} in {meta['seconds']} s")
    return 0


# --- grade ----------------------------------------------------------------------------------------
def grade(cfg, log):
    from eval_closedbook import JUDGE_PROMPT, JUDGE_SYSTEM, Grade, eval_spend
    from llm import LLM, batched_map
    rd = run_dir(cfg)
    sub = json.loads((repo_path(cfg["paths"]["splits"]) / "eval_subset.json").read_text())
    refs = {}
    for split in SPLITS:
        for r in read_jsonl(repo_path(cfg["paths"]["splits"]) / f"{split}.jsonl"):
            refs[r["qa_id"]] = r
    runs = []
    for m in MODELS:
        for arm in ARMS:
            for split in SPLITS:
                p = rd / short(m) / arm / f"{split}_predictions.jsonl"
                if p.exists():
                    runs.append((m, arm, split, {x["qa_id"]: x["prediction"] for x in read_jsonl(p)}))
    stage = f"eval_grade:{RUN}"
    judge = LLM(cfg, stage)

    def g(item):
        _, _, _, qa_id, pred = item
        r = refs[qa_id]
        return judge.complete(JUDGE_PROMPT.format(question=r["question"], reference=r["answer"], prediction=pred),
                              JUDGE_SYSTEM, role="judge", json_schema=Grade).parsed

    def items_for(k):
        out = []
        for m, arm, split, preds in runs:
            for q in sub["splits"][split][:k[split]]:
                if q in preds:
                    out.append((m, arm, split, q, preds[q]))
        return out

    full_k = {s: len(sub["splits"][s]) for s in SPLITS}
    items = items_for(full_k)
    # pilot: 50 answers to measure the actual cost per graded answer
    path = repo_path(cfg["llm"]["usage_log"])

    def spent():
        return sum(u["cost"] for u in (json.loads(x) for x in path.open(encoding="utf-8"))
                   if u["stage"] == stage and not u["cached"])
    s0 = spent()
    n0 = sum(1 for x in path.open(encoding="utf-8") if f'"{stage}"' in x and '"cached": false' in x)
    pilot = items[:50]
    batched_map(cfg, log, stage, g, pilot, workers=cfg["llm"]["standard_workers"])
    n1 = sum(1 for x in path.open(encoding="utf-8") if f'"{stage}"' in x and '"cached": false' in x)
    per = (spent() - s0) / (n1 - n0) if n1 > n0 else 0.0028
    log.info(f"grading pilot: {n1 - n0} new answers, ${per:.5f} per graded answer")
    room = cfg["budget"]["eval_max_usd"] - (eval_spend(cfg) + spent())   # all eval grading so far
    need = (len(items) - len(pilot)) * per
    k = dict(full_k)
    if need > room:
        frac = max(room / need, 0.0)
        k = {s: max(int(full_k[s] * frac), 50) for s in SPLITS}
        decision(cfg, "grading", f"subset shrunk uniformly to {k} per split (need ${need:.2f} > room ${room:.2f})",
                 "eval_max_usd; same questions for every model and arm")
        items = items_for(k)
    (rd / "grading_plan.json").write_text(json.dumps({"subset_k": k, "cost_per_answer": per, "items": len(items),
                                                      "room_usd": round(room, 2), "need_usd": round(need, 2)}, indent=2))
    res = batched_map(cfg, log, stage, g, items, workers=cfg["llm"]["standard_workers"])
    grades = defaultdict(dict)
    for i, (m, arm, split, q, _) in enumerate(items):
        if not isinstance(res[i], Exception):
            grades[(m, arm, split)][q] = {"verdict": res[i].verdict, "rationale": res[i].rationale}
    for (m, arm, split), gmap in grades.items():
        with (rd / short(m) / arm / f"{split}_grades.jsonl").open("w", encoding="utf-8") as f:
            for q, v in gmap.items():
                f.write(json.dumps({"qa_id": q, **v}, ensure_ascii=False) + "\n")
    log.info(f"graded {sum(len(v) for v in grades.values())} answers for {len(grades)} model/arm/split runs; "
             f"run grading spend ${spent():.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["subset", "download", "smoke", "epochs", "train", "generate", "grade", "report"])
    ap.add_argument("--model")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--split", choices=SPLITS + ["test_seen_facts"])
    args = ap.parse_args()
    cfg = load_config()
    log = get_logger("model_runs", cfg)
    if args.cmd == "subset":
        build_subset(cfg, log)
    elif args.cmd == "download":
        sys.exit(download(cfg, args.model, log))
    elif args.cmd == "smoke":
        sys.exit(py("src/train_qlora.py", "--model", args.model, "--max-steps", "30",
                    "--output", f"models/{short(args.model)}"))
    elif args.cmd == "epochs":
        choose_epochs(cfg, log)
    elif args.cmd == "train":
        epochs = json.loads((run_dir(cfg) / "epochs.json").read_text())["epochs"]
        sys.exit(py("src/train_qlora.py", "--model", args.model, "--epochs", str(epochs),
                    "--output", f"models/{short(args.model)}"))
    elif args.cmd == "generate":
        sys.exit(generate(cfg, args.model, args.arm, args.split, log))
    elif args.cmd == "grade":
        sys.exit(grade(cfg, log))
    elif args.cmd == "report":
        from run_results import write_all
        write_all(cfg, RUN, MODELS, ARMS, SPLITS, log)


if __name__ == "__main__":
    main()
