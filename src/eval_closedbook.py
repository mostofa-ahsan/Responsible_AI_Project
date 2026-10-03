"""Closed-book evaluation of the base or fine-tuned model on a test split.

Each test question is answered WITHOUT context, then graded against the reference answer:
  - judge (config models.judge): correct / partial / incorrect + score (1 / 0.5 / 0) and rationale
  - token F1 and ROUGE-L against the reference (local, no API)
Backends:
  --backend vllm      (default) base Qwen3-8B (+ --adapter LoRA) in vLLM, run by
                      src/vllm_generate.py under .venv-vllm (vLLM needs a newer torch than the main
                      venv); greedy, thinking disabled. Falls back to hf if vLLM is unavailable or fails.
  --backend hf        local transformers generate; --adapter adds the LoRA adapter; --load-4bit
                      loads the base in NF4 like training
  --backend endpoint  config models.closed_book_baseline (openai_compatible server)
Grading: the Opus judge via one Message Batch (--judge-mode batch, default; 50% price) or the
standard API (--judge-mode standard). Calls are cached, so re-runs are free.
Budget: in batch mode a stratified pilot of --pilot-n answers is graded first and the actual
cost per graded answer is measured from llm_usage.jsonl. If grading the rest at that rate would
exceed budget.eval_run_max_usd or what is left of budget.eval_max_usd (all eval grading so far),
a stratified sample (q_type x dimension) that fits is graded instead (the pilot answers are
always included), and the decision is logged.
Results: judge accuracy (correct = 1, partial = 0.5), token F1 and ROUGE-L with bootstrap 95%
confidence intervals, overall and by q_type and dimension.
Writes eval/<arm>/<split>_predictions.jsonl, _graded.jsonl and _summary.json.

--dry-run prints the generation and judge prompts for --limit questions and checks the vLLM
install, without loading a model or calling an API (safe on CPU while the GPU is busy).

Usage:
    python src/eval_closedbook.py --arm base --split test_heldout_docs
    python src/eval_closedbook.py --arm ft --adapter models/qwen3-8b-qlora/closedbook/final --split test_indomain
    python src/eval_closedbook.py --dry-run --limit 3
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from llm import LLM, batched_map
from qa_common import read_jsonl, run_parallel
from utils import Report, get_logger, load_config, repo_path

JUDGE_SYSTEM = """You grade answers to questions about responsible AI in higher education against \
a reference answer written from the source document. Judge factual agreement with the reference, \
not style or length.

correct   = states the key information of the reference without contradicting it (extra correct detail is fine)
partial   = gets part of the key information right, or is vague where the reference is specific
incorrect = contradicts the reference, misses the key information, or answers a different question"""

JUDGE_PROMPT = """Question: {question}

Reference answer: {reference}

Model answer: {prediction}"""


class Grade(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    rationale: str = Field(description="One sentence")


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(pred, ref):
    p, r = tokens(pred), tokens(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    prec, rec = lcs / len(p), lcs / len(r)
    return 0.0 if lcs == 0 else 2 * prec * rec / (prec + rec)


def messages_for(system, question):
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


class HFGenerator:
    def __init__(self, cfg, adapter=None, load_4bit=False, model=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        model_id = model or cfg["base_model"]
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"torch_dtype": torch.bfloat16, "device_map": {"": 0} if torch.cuda.is_available() else None}
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.torch = torch

    def generate(self, batch_messages, max_new_tokens):
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for m in batch_messages]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


VLLM_PYTHON = repo_path(".venv-vllm/bin/python")
GRADE_CALL_COST_BATCH = 0.0015      # Opus, ~400 input + ~60 output tokens, batch price (fallback estimate)


def bootstrap_ci(values, n_boot=2000, seed=0):
    """Mean with a percentile bootstrap 95% CI."""
    import random as _r
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = _r.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": round(sum(values) / n, 4), "lo": round(means[int(0.025 * n_boot)], 4),
            "hi": round(means[int(0.975 * n_boot) - 1], 4), "n": n}


def eval_spend(cfg):
    path = repo_path(cfg["llm"]["usage_log"])
    if not path.exists():
        return 0.0
    return sum(u["cost"] for u in (json.loads(line) for line in path.open(encoding="utf-8"))
               if u["stage"].startswith("eval_closedbook_judge") and not u["cached"])


def stratified(rows, k, seed=0):
    """About k rows, proportional over q_type x dimension (at least one per stratum while k allows)."""
    import random as _r
    if k >= len(rows):
        return list(rows)
    strata = {}
    for r in rows:
        strata.setdefault((r["q_type"], r["dimension"]), []).append(r)
    rng = _r.Random(seed)
    out = []
    for _, v in sorted(strata.items()):
        out += rng.sample(v, min(len(v), max(1, round(k * len(v) / len(rows)))))
    rng.shuffle(out)
    return out[:k]


def stage_cost(cfg, stage):
    """(USD, calls) of new (non-cached) calls logged for a stage."""
    path = repo_path(cfg["llm"]["usage_log"])
    rows = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    us = [u for u in rows if u["stage"] == stage and not u["cached"]]
    return sum(u["cost"] for u in us), len(us)


def grading_budget_sample(cfg, rows, log, label, per_call=GRADE_CALL_COST_BATCH, keep=()):
    """Rows to grade within the eval budget (all, or a stratified sample that includes `keep`)."""
    b = cfg.get("budget", {})
    keep_ids = {r["qa_id"] for r in keep}
    rest = [r for r in rows if r["qa_id"] not in keep_ids]
    room = min(b.get("eval_run_max_usd", float("inf")), b.get("eval_max_usd", float("inf")) - eval_spend(cfg))
    est = len(rest) * per_call
    log.info(f"grading estimate ${est:.2f} for {len(rest)} more answers at ${per_call:.5f}/answer; "
             f"budget room ${room:.2f}")
    if est <= room:
        return rows
    k = max(int(room / per_call), 0)
    out = list(keep) + stratified(rest, k)
    msg = (f"{label}: grading a stratified sample of {len(out)}/{len(rows)} "
           f"(est ${est:.2f} at ${per_call:.5f}/answer > room ${room:.2f})")
    log.warning(msg)
    dec = repo_path(cfg["paths"]["logs"]) / "UNATTENDED_DECISIONS.md"
    if dec.exists():
        import time as _t
        with dec.open("a", encoding="utf-8") as f:
            f.write(f"| {_t.strftime('%H:%M')} | eval {label} | {msg} | eval budget guard |\n")
    return out


def vllm_check():
    if not VLLM_PYTHON.exists():
        return False, "no .venv-vllm"
    r = subprocess.run([str(VLLM_PYTHON), str(repo_path("src/vllm_generate.py")), "--model", "x", "--input", "x",
                        "--output", "x", "--check"], capture_output=True, text=True)
    try:
        info = json.loads(r.stdout.strip().splitlines()[-1])
        return info["vllm_ok"], f"missing {info['missing']}" if info["missing"] else "ok"
    except Exception:  # noqa: BLE001
        return False, (r.stderr or r.stdout)[-300:]


def generate_vllm(cfg, rows, system, adapter, log, model=None):
    """Run the vLLM worker on rows; returns {qa_id: prediction}."""
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, "in.jsonl"), os.path.join(d, "out.jsonl")
        with open(inp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"qa_id": r["qa_id"], "messages": messages_for(system, r["question"])},
                                   ensure_ascii=False) + "\n")
        cmd = [str(VLLM_PYTHON), str(repo_path("src/vllm_generate.py")), "--model", model or cfg["base_model"],
               "--input", inp, "--output", out, "--max-new-tokens", str(cfg["eval"]["max_new_tokens"]),
               "--max-lora-rank", str(cfg["train"]["lora_r"])]
        if adapter:
            cmd += ["--adapter", str(repo_path(adapter))]
        log.info(f"vLLM: {len(rows)} questions{' with adapter ' + adapter if adapter else ''}")
        subprocess.run(cmd, check=True)
        return {json.loads(line)["qa_id"]: json.loads(line)["prediction"] for line in open(out, encoding="utf-8")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="base", help="label for the output folder, e.g. base, ft")
    ap.add_argument("--split", default="test_heldout_docs")
    ap.add_argument("--backend", choices=["vllm", "hf", "endpoint"], default="vllm")
    ap.add_argument("--judge-mode", choices=["batch", "standard"], default="batch")
    ap.add_argument("--pilot-n", type=int, default=50, help="answers graded first to measure cost per answer")
    ap.add_argument("--adapter", help="LoRA adapter dir (hf backend)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    log = get_logger("eval_closedbook", cfg)
    rows = read_jsonl(repo_path(cfg["paths"]["splits"]) / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"no rows in split {args.split}")
    system = cfg["train"]["system_prompt"]

    if args.dry_run:
        ok, why = vllm_check()
        print(f"=== vLLM backend available: {ok} ({why}); fallback: hf")
        for r in rows:
            print("=== generation messages:", json.dumps(messages_for(system, r["question"]), ensure_ascii=False))
            print("=== judge prompt:\n" + JUDGE_PROMPT.format(question=r["question"], reference=r["answer"],
                                                            prediction="<model answer>"))
        print(f"[dry run] {len(rows)} questions from {args.split}; no model loaded, no API called")
        return

    out_dir = repo_path(cfg["eval"]["output_dir"]) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{args.split}_predictions.jsonl"
    done = {p["qa_id"]: p for p in read_jsonl(pred_path)} if pred_path.exists() else {}
    todo = [r for r in rows if r["qa_id"] not in done]
    log.info(f"{args.arm}/{args.split}: {len(rows)} questions, {len(todo)} to answer")

    backend = args.backend
    if todo and backend == "vllm":
        ok, why = vllm_check()
        if ok:
            try:
                preds_new = generate_vllm(cfg, todo, system, args.adapter, log)
                with pred_path.open("a", encoding="utf-8") as f:
                    for r in todo:
                        if r["qa_id"] in preds_new:
                            f.write(json.dumps({"qa_id": r["qa_id"], "prediction": preds_new[r["qa_id"]]},
                                               ensure_ascii=False) + "\n")
                todo = []
            except Exception as e:  # noqa: BLE001
                log.warning(f"vLLM failed ({type(e).__name__}: {e}); falling back to HF generate")
                backend = "hf"
        else:
            log.warning(f"vLLM unavailable ({why}); falling back to HF generate")
            backend = "hf"
    if todo:
        if backend == "hf":
            gen = HFGenerator(cfg, args.adapter, args.load_4bit)
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i:i + args.batch_size]
                outs = gen.generate([messages_for(system, r["question"]) for r in batch], cfg["eval"]["max_new_tokens"])
                with pred_path.open("a", encoding="utf-8") as f:
                    for r, o in zip(batch, outs):
                        f.write(json.dumps({"qa_id": r["qa_id"], "prediction": o}, ensure_ascii=False) + "\n")
                log.info(f"answered {min(i + args.batch_size, len(todo))}/{len(todo)}")
        else:
            llm = LLM(cfg, f"eval_closedbook:{args.arm}")

            def ans(r):
                return llm.complete(r["question"], system, role="closed_book_baseline").text.strip()
            with pred_path.open("a", encoding="utf-8") as f:
                for r, out in run_parallel(ans, todo, cfg["llm"]["standard_workers"], log):
                    if not isinstance(out, Exception):
                        f.write(json.dumps({"qa_id": r["qa_id"], "prediction": out}, ensure_ascii=False) + "\n")
    preds = {p["qa_id"]: p["prediction"] for p in read_jsonl(pred_path)}

    stage = f"eval_closedbook_judge:{args.arm}:{args.split}"
    judge = LLM(cfg, stage)

    def grade(r):
        return judge.complete(JUDGE_PROMPT.format(question=r["question"], reference=r["answer"],
                                                  prediction=preds[r["qa_id"]]), JUDGE_SYSTEM, role="judge",
                              json_schema=Grade).parsed
    answered = [r for r in rows if r["qa_id"] in preds]
    pilot, per_call = [], GRADE_CALL_COST_BATCH
    if args.judge_mode == "batch" and len(answered) > args.pilot_n:
        pilot = stratified(answered, args.pilot_n, seed=1)
        before_cost, before_n = stage_cost(cfg, stage)
        batched_map(cfg, log, stage, grade, pilot, workers=cfg["llm"]["standard_workers"])
        after_cost, after_n = stage_cost(cfg, stage)
        if after_n > before_n:
            per_call = (after_cost - before_cost) / (after_n - before_n)
            log.info(f"grading pilot: {after_n - before_n} answers cost ${after_cost - before_cost:.4f} "
                     f"-> ${per_call:.5f} per graded answer (measured)")
        else:   # pilot already cached from an earlier run: use the stage's logged average
            per_call = (after_cost / after_n) if after_n else GRADE_CALL_COST_BATCH
            log.info(f"grading pilot cached; using logged ${per_call:.5f} per graded answer")
    to_grade = grading_budget_sample(cfg, answered, log, f"{args.arm}/{args.split}", per_call=per_call, keep=pilot)
    graded = {}
    if args.judge_mode == "batch":
        res = batched_map(cfg, log, stage, grade, to_grade, workers=cfg["llm"]["standard_workers"])
        for i, r in enumerate(to_grade):
            if not isinstance(res[i], Exception):
                graded[r["qa_id"]] = res[i]
    else:
        for r, g in run_parallel(grade, to_grade, cfg["llm"]["max_workers"], log):
            if not isinstance(g, Exception):
                graded[r["qa_id"]] = g
    score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    results = []
    for r in rows:
        if r["qa_id"] not in graded:
            continue
        p = preds[r["qa_id"]]
        results.append({"qa_id": r["qa_id"], "q_type": r["q_type"], "dimension": r["dimension"],
                        "difficulty": r["difficulty"], "question": r["question"], "reference": r["answer"],
                        "prediction": p, "verdict": graded[r["qa_id"]].verdict,
                        "judge_score": score[graded[r["qa_id"]].verdict],
                        "judge_rationale": graded[r["qa_id"]].rationale,
                        "token_f1": round(token_f1(p, r["answer"]), 4), "rouge_l": round(rouge_l(p, r["answer"]), 4)})
    with (out_dir / f"{args.split}_graded.jsonl").open("w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    metrics = ("judge_score", "token_f1", "rouge_l")

    def block(sel):
        return {m: bootstrap_ci([x[m] for x in sel]) for m in metrics}
    summary = {"arm": args.arm, "split": args.split, "n_questions": len(rows), "n_graded": len(results),
               "grading_cost_per_answer_usd": round(per_call, 6),
               "verdicts": dict(Counter(x["verdict"] for x in results)), "overall": block(results),
               "by_q_type": {t: block([x for x in results if x["q_type"] == t])
                             for t in sorted({x["q_type"] for x in results})},
               "by_dimension": {d: block([x for x in results if x["dimension"] == d])
                                for d in sorted({x["dimension"] for x in results})}}
    (out_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    rep = Report(f"eval_closedbook_{args.arm}_{args.split}", cfg)
    rep(f"=== Closed-book eval: {args.arm} on {args.split} ({len(results)} graded of {len(rows)}) ===")
    rep(f"verdicts: {summary['verdicts']}")

    def fmt(b, m):
        x = b[m]
        return f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]" if x["mean"] is not None else "-"
    rep(f"\n{'group':28s} {'n':>5s}  {'judge accuracy':>22s}  {'token F1':>22s}  {'ROUGE-L':>22s}")
    rep(f"{'overall':28s} {len(results):>5d}  " + "  ".join(f"{fmt(summary['overall'], m):>22s}" for m in metrics))
    for title, key in (("q_type", "by_q_type"), ("dimension", "by_dimension")):
        rep(f"-- by {title}")
        for g, b in summary[key].items():
            rep(f"  {g:26s} {b['judge_score']['n']:>5d}  " + "  ".join(f"{fmt(b, m):>22s}" for m in metrics))
    rep("(bootstrap 95% CIs, 2,000 resamples; judge accuracy: correct = 1, partial = 0.5, incorrect = 0)")
    rep.save()


if __name__ == "__main__":
    main()
