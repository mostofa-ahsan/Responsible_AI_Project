"""Phase 4 stage 3: filter generated QA pairs and write the final pilot files.

Filters (each recorded; a pair passes only if all apply-able filters pass):
  1. grounding  - judge model scores how well the evidence supports the answer (1-5); keep >= min_grounding
  2. standalone - judge model: is the question understandable without the source? (yes/no)
  3. format     - answer body is 2-5 sentences and the question has exactly 2 paraphrases
  4. dedup      - question embedding cosine > dedup_cosine to an already-kept pair -> drop
Deferred until the index and the local base model exist (recorded as null):
  closed-book check (closed_book_correct) and the retrieval filter.

The judge also reports whether the paraphrases keep the question's meaning
(recorded in judge_scores, not used as a filter in the pilot).

Writes data/qa_pairs/<run>.jsonl, data/qa_pairs/<run>_review.csv (with an empty
human_verdict column) and the pilot report. Judge calls are cached, so re-runs
only pay for new pairs.

Usage:
    python src/qa_filter.py --pilot [--limit N] [--seed 0]
"""

import argparse
import csv
import json
import random
import re
from collections import Counter

from pydantic import BaseModel, Field

from llm import LLM
from qa_common import read_jsonl, run_parallel
from utils import Report, get_logger, load_config, repo_path

SYSTEM = """You are a strict reviewer of a question-answering dataset built from books and \
journal articles about responsible AI in higher education. You judge each pair only against the \
evidence quoted from the source; you do not use outside knowledge to fill gaps.

grounding_score (1-5): how fully the evidence supports every claim in the answer.
  5 = every statement is directly stated in the evidence
  4 = fully supported; at most trivial rewording or an obvious inference
  3 = mostly supported, but one statement goes beyond the evidence or adds a detail not in it
  2 = several unsupported statements, or an important distortion
  1 = largely unsupported or contradicted by the evidence
Ignore the trailing citation in parentheses when scoring.

standalone: true if someone who has never seen the source understands exactly what the question
asks (no "this study", "the chapter", "the author", undefined acronyms or references to unseen
context). A question may name a specific study, book or framework by its content.

paraphrases_equivalent: true if both paraphrases ask for the same information as the question."""

PROMPT = """Question: {question}
Paraphrases:
1. {p1}
2. {p2}

Answer: {answer}

Evidence quoted from the source:
<evidence>
{evidence}
</evidence>"""


class Judgement(BaseModel):
    grounding_score: int = Field(ge=1, le=5)
    grounding_rationale: str = Field(description="One or two sentences naming any unsupported statement")
    standalone: bool
    standalone_rationale: str
    paraphrases_equivalent: bool


SENT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def answer_sentences(answer):
    body = re.sub(r"\s*\([^()]*,\s*pp?\.\s*[\d\-–]+\)\s*$", "", answer).strip()
    return len([s for s in SENT_RX.split(body) if s.strip()])


def dedup(pairs, cfg, threshold, log):
    """Mark pairs whose question is a near-duplicate (cosine > threshold) of an earlier kept pair."""
    from embed import encode, load_model
    keep_idx = [i for i, p in enumerate(pairs) if p["_ok"]]
    if not keep_idx:
        return
    model = load_model(cfg["embed"])
    vecs = encode(model, [pairs[i]["question"] for i in keep_idx], cfg["embed"]["batch_size"])
    kept = []
    for j, i in enumerate(keep_idx):
        sims = [(float(vecs[j] @ vecs[k]), pairs[keep_idx[k]]["qa_id"]) for k in kept]
        best = max(sims, default=(0.0, None))
        pairs[i]["judge_scores"]["max_question_cosine"] = round(best[0], 3)
        if best[0] > threshold:
            pairs[i]["_fail"].append(f"duplicate(cos={best[0]:.3f} of {best[1]})")
            pairs[i]["_ok"] = False
        else:
            kept.append(j)
    log.info(f"dedup: {len(keep_idx) - len(kept)} near-duplicates removed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, help="only judge the first N pairs")
    ap.add_argument("--seed", type=int, default=0, help="seed for report examples")
    ap.add_argument("--skip-dedup", action="store_true",
                    help="judge only (e.g. while the GPU is busy); re-run later, judge calls are cached")
    args = ap.parse_args()
    if not args.pilot:
        ap.error("only --pilot is implemented (the full run is Phase 5)")

    cfg = load_config()
    qcfg = cfg["qa"]
    log = get_logger("qa_filter", cfg)
    out_dir = repo_path(cfg["paths"]["qa_pairs"])
    run = "pilot"
    gen = read_jsonl(out_dir / f"{run}_generated.jsonl")
    if args.limit:
        gen = gen[:args.limit]
    llm = LLM(cfg, "qa_filter")

    def work(g):
        p = g["paraphrases"] + ["", ""]
        prompt = PROMPT.format(question=g["question"], p1=p[0], p2=p[1], answer=g["answer"],
                               evidence=g["evidence"])
        return llm.complete(prompt, SYSTEM, role="judge", json_schema=Judgement)

    judged = {}
    for g, res in run_parallel(work, gen, cfg["llm"]["max_workers"], log):
        judged[g["qa_id"]] = res

    pairs = []
    for g in gen:
        res = judged[g["qa_id"]]
        p = {**g, "judge_scores": {}, "_fail": [], "_ok": True}
        if isinstance(res, Exception):
            p["_fail"].append(f"judge_error({res})")
        else:
            j = res.parsed
            p["judge_scores"] = {"grounding": j.grounding_score, "grounding_rationale": j.grounding_rationale,
                                 "standalone": j.standalone, "standalone_rationale": j.standalone_rationale,
                                 "paraphrases_equivalent": j.paraphrases_equivalent,
                                 "judge_model": res.served_model}
            if j.grounding_score < qcfg["min_grounding"]:
                p["_fail"].append(f"grounding({j.grounding_score})")
            if not j.standalone:
                p["_fail"].append("not_standalone")
        n_sent = answer_sentences(g["answer"])
        p["judge_scores"]["answer_sentences"] = n_sent
        if not 2 <= n_sent <= 5 or len(g["paraphrases"]) != 2:
            p["_fail"].append(f"format(sentences={n_sent},paraphrases={len(g['paraphrases'])})")
        p["_ok"] = not p["_fail"]
        pairs.append(p)

    if args.skip_dedup:
        log.warning("dedup skipped (--skip-dedup); re-run without it before using the output")
    else:
        dedup(pairs, cfg, qcfg["dedup_cosine"], log)

    final = []
    for p in pairs:
        row = {
            "qa_id": p["qa_id"], "group_id": p["group_id"], "question": p["question"],
            "paraphrases": p["paraphrases"], "answer": p["answer"], "evidence": p["evidence"],
            "doc_id": p["doc_id"], "section_path": p["section_path"], "page": p["page"],
            "q_type": p["q_type"], "dimension": p["dimension"], "difficulty": p["difficulty"],
            "closed_book_correct": None,          # pending: needs the local base model
            "retrieval_rank": None,               # pending: retrieval filter after the index exists
            "judge_scores": p["judge_scores"], "passed_filters": p["_ok"],
            "reject_reason": ";".join(p["_fail"]),
            "chunk_id": p["chunk_id"], "unit_ids": p["unit_ids"], "generator": p["generator"],
        }
        final.append(row)
    with (out_dir / f"{run}.jsonl").open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / f"{run}_review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qa_id", "passed_filters", "reject_reason", "q_type", "dimension", "difficulty",
                    "question", "paraphrase_1", "paraphrase_2", "answer", "evidence",
                    "grounding", "standalone", "paraphrases_equivalent", "judge_rationale",
                    "human_verdict", "human_notes"])
        for r in final:
            js = r["judge_scores"]
            para = r["paraphrases"] + ["", ""]
            w.writerow([r["qa_id"], r["passed_filters"], r["reject_reason"], r["q_type"], r["dimension"],
                        r["difficulty"], r["question"], para[0], para[1], r["answer"], r["evidence"],
                        js.get("grounding"), js.get("standalone"), js.get("paraphrases_equivalent"),
                        js.get("grounding_rationale", ""), "", ""])

    report(final, cfg, llm, args.seed, run)


def report(final, cfg, llm, seed, run):
    rep = Report("qa_pilot", cfg)
    n = len(final)
    rep(f"=== Phase 4a pilot: {n} QA pairs from {len({r['chunk_id'] for r in final})} chunks, "
        f"{len({r['doc_id'] for r in final})} docs ===")
    rep("single-chunk types only; closed-book check and retrieval filter pending (null in output)")
    passed = [r for r in final if r["passed_filters"]]
    rep(f"passed all filters: {len(passed)}/{n} ({len(passed) / max(n, 1):.0%})")

    def rate(pred):
        k = sum(pred(r) for r in final)
        return f"{k}/{n} ({k / max(n, 1):.0%})"
    rep("\n-- Pass rate per filter (each judged independently) --")
    rep(f"grounding >= {cfg['qa']['min_grounding']}: "
        f"{rate(lambda r: (r['judge_scores'].get('grounding') or 0) >= cfg['qa']['min_grounding'])}")
    rep(f"  grounding score distribution: {dict(sorted(Counter(r['judge_scores'].get('grounding') for r in final).items(), key=lambda kv: (kv[0] is None, kv[0])))}")
    rep(f"standalone: {rate(lambda r: r['judge_scores'].get('standalone') is True)}")
    rep(f"format (2-5 sentences, 2 paraphrases): {rate(lambda r: 'format(' not in r['reject_reason'])}")
    rep(f"dedup (not a near-duplicate, cos <= {cfg['qa']['dedup_cosine']}): "
        f"{rate(lambda r: 'duplicate(' not in r['reject_reason'])}")
    rep(f"(recorded only) paraphrases equivalent: {rate(lambda r: r['judge_scores'].get('paraphrases_equivalent') is True)}")
    rep(f"reject reasons: {dict(Counter(x.split('(')[0] for r in final for x in r['reject_reason'].split(';') if x))}")

    for key in ("q_type", "dimension", "difficulty"):
        g, p = Counter(r[key] for r in final), Counter(r[key] for r in passed)
        rep(f"\n-- {key}: generated -> passed --")
        for k, v in g.most_common():
            rep(f"  {k:24s} {v:4d} -> {p[k]:4d}")
    rep(f"\nper doc (generated -> passed): " + ", ".join(
        f"{d}: {sum(r['doc_id'] == d for r in final)}->{sum(r['doc_id'] == d for r in passed)}"
        for d in sorted({r['doc_id'] for r in final})))

    usage = [json.loads(line) for line in repo_path(cfg["llm"]["usage_log"]).open(encoding="utf-8")]
    stages = ("qa_extract", "qa_generate", "qa_filter")
    rep("\n-- Cost (all pilot stages, from logs/llm_usage.jsonl; cached re-runs cost $0) --")
    total = 0.0
    for st in stages:
        rows = [u for u in usage if u["stage"] == st and not u["cached"]]
        c = sum(u["cost"] for u in rows)
        total += c
        rep(f"  {st:12s} calls={len(rows):4d}  in={sum(u['input'] for u in rows):>9,}  "
            f"out={sum(u['output'] for u in rows):>9,}  ${c:.4f}")
    rep(f"  {'TOTAL':12s} ${total:.4f}   (${total / max(n, 1):.4f} per generated pair, "
        f"${total / max(len(passed), 1):.4f} per passed pair)")
    rep("this run (stage 3):")
    for line in llm.summary():
        rep(line)

    rng = random.Random(seed)
    rejected = [r for r in final if not r["passed_filters"]]

    def show(r):
        js = r["judge_scores"]
        rep(f"\n[{r['qa_id']}] {r['q_type']} | {r['dimension']} | {r['difficulty']} | "
            f"grounding={js.get('grounding')} standalone={js.get('standalone')}"
            + (f" | REJECTED: {r['reject_reason']}" if r["reject_reason"] else ""))
        rep(f"Q: {r['question']}")
        rep(f"A: {r['answer']}")
        if r["reject_reason"]:
            rep(f"judge: {js.get('grounding_rationale', '')} | {js.get('standalone_rationale', '')}")
    rep("\n-- 5 random passed examples --")
    for r in rng.sample(passed, min(5, len(passed))):
        show(r)
    rep("\n-- 5 random rejected examples --")
    for r in rng.sample(rejected, min(5, len(rejected))):
        show(r)
    rep.save()


if __name__ == "__main__":
    main()
