"""Closed-book evaluation of the base or fine-tuned model on a test split.

Each test question is answered WITHOUT context, then graded against the reference answer:
  - judge (config models.judge): correct / partial / incorrect + score (1 / 0.5 / 0) and rationale
  - token F1 and ROUGE-L against the reference (local, no API)
Backends:
  --backend hf        local transformers; --adapter <dir> adds a LoRA adapter (fine-tuned arm);
                      --load-4bit loads the base in NF4 like training
  --backend endpoint  config models.closed_book_baseline (openai_compatible, e.g. vLLM serving the
                      base model and the adapter), thinking disabled via chat_template_kwargs
Writes eval/<arm>/<split>_predictions.jsonl and eval/<arm>/<split>_summary.json. Judge calls go
through the LLM cache (re-runs are free; in collect mode they can be sent as a batch).

--dry-run prints the generation and judge prompts for --limit questions without loading a model
or calling an API (safe on CPU while the GPU is busy).

Usage:
    python src/eval_closedbook.py --arm base --split test_heldout_docs --backend hf
    python src/eval_closedbook.py --arm ft --adapter models/qwen3-8b-qlora/closedbook/final --split test_indomain
    python src/eval_closedbook.py --dry-run --limit 3
"""

import argparse
import json
import re
import statistics
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from llm import LLM
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
    def __init__(self, cfg, adapter=None, load_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(cfg["base_model"], padding_side="left")
        kw = {"torch_dtype": torch.bfloat16, "device_map": {"": 0} if torch.cuda.is_available() else None}
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kw)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="base", help="label for the output folder, e.g. base, ft")
    ap.add_argument("--split", default="test_heldout_docs")
    ap.add_argument("--backend", choices=["hf", "endpoint"], default="hf")
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

    if todo:
        if args.backend == "hf":
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

    judge = LLM(cfg, f"eval_closedbook_judge:{args.arm}")

    def grade(r):
        return judge.complete(JUDGE_PROMPT.format(question=r["question"], reference=r["answer"],
                                                  prediction=preds[r["qa_id"]]), JUDGE_SYSTEM, role="judge",
                              json_schema=Grade).parsed
    graded = {}
    for r, g in run_parallel(grade, [r for r in rows if r["qa_id"] in preds], cfg["llm"]["max_workers"], log):
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
    summary = {"arm": args.arm, "split": args.split, "n": len(results),
               "judge_score": statistics.mean(x["judge_score"] for x in results) if results else None,
               "verdicts": dict(Counter(x["verdict"] for x in results)),
               "token_f1": statistics.mean(x["token_f1"] for x in results) if results else None,
               "rouge_l": statistics.mean(x["rouge_l"] for x in results) if results else None,
               "by_q_type": {t: statistics.mean(x["judge_score"] for x in results if x["q_type"] == t)
                             for t in sorted({x["q_type"] for x in results})}}
    (out_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    rep = Report(f"eval_closedbook_{args.arm}_{args.split}", cfg)
    rep(json.dumps(summary, indent=2))
    rep.save()


if __name__ == "__main__":
    main()
