"""Phase 4 stage 3: filter (with one repair pass) and write the final run files.

Per pair:
  1. pre-filter  - banned-phrase regex (qa.banned_phrases) on question and paraphrases; a hit in
                   the question fails standalone without a judge call
  2. judge       - grounding score 1-5 (keep >= min_grounding; inference/synthesis beyond the
                   evidence caps at 3), standalone yes/no, per-paraphrase equivalence
  3. format      - answer sentences within qa.sentence_range[q_type]
  4. repair      - pairs failing grounding or standalone go back to the generator once with the
                   reason, then are re-checked (1-3); tracked separately
  5. paraphrases - non-equivalent or banned paraphrases are regenerated once and re-checked;
                   still-bad ones are dropped (the pair is not rejected)
  6. dedup       - question cosine > dedup_cosine to an already-kept pair -> drop
Deferred (null): closed_book_correct and retrieval_rank.

Writes data/qa_pairs/<run>.jsonl and <run>_review.csv (filled human_verdict/human_notes in an
existing review file are preserved), and logs/qa_<run>_report.txt with a comparison to v1.
All LLM calls are cached.

Usage:
    python src/qa_filter.py --run pilot_v2 [--limit N] [--skip-dedup] [--seed 0]
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from llm import LLM
from qa_common import (banned_regex, citation_title, load_chunks, load_metadata, read_jsonl,
                       run_parallel, run_paths, write_review_csv)
from qa_generate import regenerate_paraphrases, repair_pair, to_row
from utils import Report, get_logger, load_config, repo_path

SYSTEM = """You are a strict reviewer of a question-answering dataset built from books and \
journal articles about responsible AI in higher education. You judge each pair only against the \
evidence quoted from the source; you do not use outside knowledge to fill gaps.

grounding_score (1-5): does the answer restate only what the evidence says?
  5 = every statement is directly stated in the evidence
  4 = fully supported; only trivial rewording
  3 = adds an inference, synthesis, interpretation or connective claim the evidence does not
      state (e.g. "this shows that ...", combining facts into a new conclusion), or one detail
      not in the evidence
  2 = several unsupported statements, or an important distortion
  1 = largely unsupported or contradicted by the evidence

standalone: true if someone who has never seen the source understands exactly what the question
asks: no "the evidence", "the text", "this study/chapter", "the authors", "the recommendations",
undefined acronyms or references to unseen context. A question may name a specific study, book
or framework by its content.

paraphrase_equivalent: for each paraphrase in order, true only if it asks for exactly the same
information as the question (same scope, entities and expected answer) and is itself standalone."""

PROMPT = """Question: {question}
Paraphrases:
{paraphrases}

Answer: {answer}

Evidence quoted from the source:
<evidence>
{evidence}
</evidence>"""

PARA_SYSTEM = """You check paraphrases of a question for a QA dataset. A paraphrase is equivalent \
only if it asks for exactly the same information as the question (same scope, entities and \
expected answer) and is standalone (no "the text", "this study", "the authors", etc.)."""

PARA_PROMPT = """Question: {question}

Candidate paraphrases:
{cands}"""


class Judgement(BaseModel):
    grounding_score: int = Field(ge=1, le=5)
    grounding_rationale: str = Field(description="One or two sentences naming any unsupported statement")
    standalone: bool
    standalone_rationale: str
    paraphrase_equivalent: list[bool] = Field(description="One boolean per paraphrase, in order")


class ParaCheck(BaseModel):
    equivalent: list[bool] = Field(description="One boolean per candidate, in order")


SENT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def sentences(text):
    return len([s for s in SENT_RX.split(text.strip()) if s.strip()])


class Checker:
    def __init__(self, cfg, llm):
        self.cfg, self.llm = cfg, llm
        self.banned = banned_regex(cfg)
        self.qcfg = cfg["qa"]

    def check(self, p):
        """Pre-filter, judge and format-check one pair. Returns (fails, scores)."""
        fails, scores = [], {}
        m = self.banned.search(p["question"])
        scores["prefilter_question"] = m.group(0) if m else None
        scores["prefilter_paraphrases"] = [bool(self.banned.search(x)) for x in p["paraphrases"]]
        if m:
            fails.append(f"not_standalone(prefilter:'{m.group(0)}')")
        else:
            paras = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(p["paraphrases"])) or "(none)"
            j = self.llm.complete(PROMPT.format(question=p["question"], paraphrases=paras,
                                                answer=p["answer"], evidence=p["evidence"]),
                                  SYSTEM, role="judge", json_schema=Judgement)
            jp = j.parsed
            eq = (jp.paraphrase_equivalent + [False] * len(p["paraphrases"]))[:len(p["paraphrases"])]
            scores.update(grounding=jp.grounding_score, grounding_rationale=jp.grounding_rationale,
                          standalone=jp.standalone, standalone_rationale=jp.standalone_rationale,
                          paraphrase_equivalent=eq, judge_model=j.served_model)
            if jp.grounding_score < self.qcfg["min_grounding"]:
                fails.append(f"grounding({jp.grounding_score})")
            if not jp.standalone:
                fails.append("not_standalone(judge)")
        lo, hi = self.qcfg["sentence_range"][p["q_type"]]
        n = sentences(p["answer"])
        scores["answer_sentences"] = n
        if not lo <= n <= hi:
            fails.append(f"format(sentences={n},allowed={lo}-{hi})")
        return fails, scores

    def fix_paraphrases(self, p, scores):
        """Regenerate non-equivalent/banned paraphrases once; drop the ones still bad."""
        eq = scores.get("paraphrase_equivalent", [True] * len(p["paraphrases"]))
        pre = scores.get("prefilter_paraphrases", [False] * len(p["paraphrases"]))
        good = [x for x, e, b in zip(p["paraphrases"], eq, pre) if e and not b]
        bad = [x for x, e, b in zip(p["paraphrases"], eq, pre) if not (e and not b)]
        info = {"initial_bad": len(bad), "regenerated": 0, "dropped": 0}
        if not bad:
            return good, info
        new = regenerate_paraphrases(self.llm, self.cfg, p["question"], bad, len(bad)).parsed.paraphrases
        new = [x.strip() for x in new if x.strip() and not self.banned.search(x)][:len(bad)]
        if new:
            chk = self.llm.complete(PARA_PROMPT.format(question=p["question"], cands="\n".join(
                f"{i + 1}. {x}" for i, x in enumerate(new))), PARA_SYSTEM, role="judge",
                json_schema=ParaCheck).parsed.equivalent
            kept = [x for x, ok in zip(new, chk + [False] * len(new)) if ok]
        else:
            kept = []
        info["regenerated"] = len(kept)
        info["dropped"] = len(bad) - len(kept)
        return good + kept, info


def dedup(rows, cfg, threshold, log):
    from embed import encode, load_model
    idx = [i for i, r in enumerate(rows) if r["passed_filters"]]
    if not idx:
        return 0
    model = load_model(cfg["embed"])
    vecs = encode(model, [rows[i]["question"] for i in idx], cfg["embed"]["batch_size"])
    kept, removed = [], 0
    for j, i in enumerate(idx):
        best = max(((float(vecs[j] @ vecs[k]), rows[idx[k]]["qa_id"]) for k in kept), default=(0.0, None))
        rows[i]["judge_scores"]["max_question_cosine"] = round(best[0], 3)
        if best[0] > threshold:
            rows[i]["passed_filters"] = False
            rows[i]["reject_reason"] = f"duplicate(cos={best[0]:.3f} of {best[1]})"
            removed += 1
        else:
            kept.append(j)
    log.info(f"dedup: {removed} near-duplicates removed")
    return removed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="pilot_v2")
    ap.add_argument("--limit", type=int, help="only process the first N pairs")
    ap.add_argument("--skip-dedup", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("qa_filter", cfg)
    paths = run_paths(cfg, args.run)
    gen = read_jsonl(paths["generated"])
    if args.limit:
        gen = gen[:args.limit]
    sel = json.loads(paths["chunks"].read_text())
    chunks, meta = load_chunks(cfg), load_metadata(cfg)
    units = defaultdict(dict)
    for u in read_jsonl(paths["units"]):
        if u["evidence_valid"]:
            units[u["chunk_id"]][u["unit_id"].split(":")[-1]] = u
    llm = LLM(cfg, f"qa_filter:{args.run}")
    checker = Checker(cfg, llm)

    def process(g):
        fails, scores = checker.check(g)
        rec = {"pair": g, "first_fails": fails, "first_scores": scores, "repair": None}
        repairable = [f for f in fails if f.startswith(("grounding", "not_standalone"))]
        if repairable and cfg["qa"]["repair_rounds"] > 0:
            reason = "; ".join(repairable) + ". " + " ".join(
                x for x in (scores.get("grounding_rationale"), scores.get("standalone_rationale")) if x)
            fixed = repair_pair(llm, cfg, g, units[g["chunk_id"]], reason)
            c = chunks[g["chunk_id"]]
            new = to_row(fixed.parsed, g["chunk_id"], int(g["qa_id"].rsplit("q", 1)[1]), c,
                         units[g["chunk_id"]], citation_title(meta[c["doc_id"]]), None, fixed.served_model)
            new["requested"] = g["requested"]
            if not new["unit_ids"]:            # repair cited no known unit: keep original evidence
                new.update(evidence=g["evidence"], unit_ids=g["unit_ids"], citation=g["citation"])
            f2, s2 = checker.check(new)
            rec.update(pair=new, repair={"reason": reason, "fails_after": f2}, final_fails=f2, final_scores=s2)
        else:
            rec.update(final_fails=fails, final_scores=scores)
        if not rec["final_fails"]:
            paras, info = checker.fix_paraphrases(rec["pair"], rec["final_scores"])
            rec["pair"] = {**rec["pair"], "paraphrases": paras}
            rec["paraphrase_fix"] = info
        return rec

    recs = {}
    for g, res in run_parallel(process, gen, cfg["llm"]["max_workers"], log):
        recs[g["qa_id"]] = res if not isinstance(res, Exception) else {
            "pair": g, "first_fails": [f"error({res})"], "first_scores": {}, "repair": None,
            "final_fails": [f"error({res})"], "final_scores": {}}

    rows = []
    for g in gen:
        r = recs[g["qa_id"]]
        p = r["pair"]
        rows.append({
            "qa_id": p["qa_id"], "group_id": p["group_id"], "question": p["question"],
            "paraphrases": p["paraphrases"], "answer": p["answer"], "citation": p["citation"],
            "evidence": p["evidence"], "doc_id": p["doc_id"], "section_path": p["section_path"],
            "page": p["citation"]["pages"], "q_type": p["q_type"], "dimension": p["dimension"],
            "difficulty": p["difficulty"],
            "closed_book_correct": None, "retrieval_rank": None,   # pending phases
            "judge_scores": r["final_scores"], "passed_filters": not r["final_fails"],
            "reject_reason": ";".join(r["final_fails"]),
            "first_pass": {"passed": not r["first_fails"], "reject_reason": ";".join(r["first_fails"]),
                           "judge_scores": r["first_scores"]},
            "repair": None if r["repair"] is None else {
                "attempted": True, "reason": r["repair"]["reason"],
                "original_question": g["question"], "original_answer": g["answer"],
                "passed_after_repair": not r["final_fails"]},
            "paraphrase_fix": r.get("paraphrase_fix"),
            "chunk_id": p["chunk_id"], "origin": sel["origin"].get(p["chunk_id"]),
            "unit_ids": p["unit_ids"], "requested": p.get("requested"), "generator": p["generator"],
        })
    if not args.skip_dedup:
        dedup(rows, cfg, cfg["qa"]["dedup_cosine"], log)

    with paths["final"].open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    header = ["qa_id", "origin", "passed_filters", "reject_reason", "repaired", "q_type", "dimension",
              "difficulty", "question", "paraphrase_1", "paraphrase_2", "answer", "citation", "evidence",
              "grounding", "standalone", "judge_rationale"]
    review = []
    for r in rows:
        js, para = r["judge_scores"], r["paraphrases"] + ["", ""]
        pages = r["citation"]["pages"]
        review.append({
            "qa_id": r["qa_id"], "origin": r["origin"], "passed_filters": r["passed_filters"],
            "reject_reason": r["reject_reason"], "repaired": bool(r["repair"]), "q_type": r["q_type"],
            "dimension": r["dimension"], "difficulty": r["difficulty"], "question": r["question"],
            "paraphrase_1": para[0], "paraphrase_2": para[1], "answer": r["answer"],
            "citation": f"{r['citation']['title']}, p{'p' if len(pages) > 1 else ''}. "
                        f"{'-'.join(str(x) for x in (pages[0], pages[-1]) if x) if len(pages) > 1 else pages[0]}",
            "evidence": r["evidence"], "grounding": js.get("grounding"), "standalone": js.get("standalone"),
            "judge_rationale": " | ".join(x for x in (js.get("grounding_rationale"), js.get("standalone_rationale"),
                                                      f"prefilter: {js['prefilter_question']}" if js.get("prefilter_question") else None) if x),
        })
    kept_human = write_review_csv(paths["review"], header, review)
    log.info(f"wrote {paths['final'].name} and {paths['review'].name} (kept {kept_human} human verdicts)")
    report(rows, cfg, args, llm)


def pct(a, b):
    return f"{a}/{b} ({a / max(b, 1):.0%})"


def report(rows, cfg, args, llm):
    rep = Report(f"qa_{args.run}", cfg)
    n = len(rows)
    passed = [r for r in rows if r["passed_filters"]]
    first_pass = [r for r in rows if r["first_pass"]["passed"]]
    repaired = [r for r in rows if r["repair"]]
    rep(f"=== Phase 4a {args.run}: {n} QA pairs from {len({r['chunk_id'] for r in rows})} chunks, "
        f"{len({r['doc_id'] for r in rows})} docs ===")
    rep("single-chunk types only; closed_book_correct and retrieval_rank pending (null)")
    rep(f"passed (final): {pct(len(passed), n)}   first pass (before repair): {pct(len(first_pass), n)}")
    rep(f"repair: attempted {len(repaired)}, passed after repair "
        f"{pct(sum(r['repair']['passed_after_repair'] for r in repaired), len(repaired))}")

    fp = [r["first_pass"] for r in rows]
    rep("\n-- First-pass rates per filter (comparable to v1) --")
    rep(f"prefilter (banned phrase in question): "
        f"{sum(bool(f['judge_scores'].get('prefilter_question')) for f in fp)} caught before the judge")
    judged = [f for f in fp if "grounding" in f["judge_scores"]]
    rep(f"grounding >= {cfg['qa']['min_grounding']}: "
        f"{pct(sum(f['judge_scores']['grounding'] >= cfg['qa']['min_grounding'] for f in judged), len(judged))} "
        f"(of judged); distribution {dict(sorted(Counter(f['judge_scores']['grounding'] for f in judged).items()))}")
    rep(f"standalone (prefilter + judge): {pct(sum('not_standalone' not in f['reject_reason'] for f in fp), n)}")
    rep(f"format (type-specific sentence range): {pct(sum('format(' not in f['reject_reason'] for f in fp), n)}")
    eqs = [e for f in judged for e in f["judge_scores"].get("paraphrase_equivalent", [])]
    rep(f"paraphrases equivalent (first pass, per paraphrase): {pct(sum(eqs), len(eqs))}")
    pf = [r["paraphrase_fix"] for r in rows if r.get("paraphrase_fix")]
    rep(f"paraphrase repair on passing pairs: {sum(x['initial_bad'] for x in pf)} bad -> "
        f"{sum(x['regenerated'] for x in pf)} regenerated, {sum(x['dropped'] for x in pf)} dropped; "
        f"passing pairs with 2/1/0 paraphrases: "
        f"{sum(len(r['paraphrases']) == 2 for r in passed)}/{sum(len(r['paraphrases']) == 1 for r in passed)}/"
        f"{sum(len(r['paraphrases']) == 0 for r in passed)}")
    rep(f"final reject reasons: {dict(Counter(x.split('(')[0] for r in rows for x in r['reject_reason'].split(';') if x))}")

    for key in ("q_type", "difficulty", "dimension"):
        g, p = Counter(r[key] for r in rows), Counter(r[key] for r in passed)
        rep(f"\n-- {key}: generated -> passed --")
        for k, v in g.most_common():
            rep(f"  {k:24s} {v:4d} ({v / n:.0%}) -> {p[k]:4d}")

    # v1 comparison on the same chunks
    v1_path = run_paths(cfg, "pilot")["final"]
    if v1_path.exists():
        v1 = read_jsonl(v1_path)
        same = {r["chunk_id"] for r in v1}
        v2s = [r for r in rows if r["chunk_id"] in same]
        v2p = [r for r in v2s if r["passed_filters"]]
        v1p = [r for r in v1 if r["passed_filters"]]
        rep("\n-- v1 vs v2 on the same 20 chunks --")
        rep(f"{'':34s} {'v1':>14s} {'v2':>14s}")
        def line(name, a, b):
            rep(f"{name:34s} {a:>14s} {b:>14s}")
        line("pairs generated", str(len(v1)), str(len(v2s)))
        line("passed (final)", pct(len(v1p), len(v1)), pct(len(v2p), len(v2s)))
        line("passed first pass (no repair)", pct(len(v1p), len(v1)),
             pct(sum(r["first_pass"]["passed"] for r in v2s), len(v2s)))
        g1 = [r["judge_scores"].get("grounding") or 0 for r in v1]
        g2 = [r["first_pass"]["judge_scores"].get("grounding") for r in v2s if "grounding" in r["first_pass"]["judge_scores"]]
        line("grounding >= 4 (first pass)", pct(sum(x >= 4 for x in g1), len(g1)), pct(sum(x >= 4 for x in g2), len(g2)))
        line("grounding = 5 (first pass)", pct(sum(x == 5 for x in g1), len(g1)), pct(sum(x == 5 for x in g2), len(g2)))
        line("standalone (first pass)", pct(sum(r["judge_scores"].get("standalone") is True for r in v1), len(v1)),
             pct(sum("not_standalone" not in r["first_pass"]["reject_reason"] for r in v2s), len(v2s)))
        line("passing pairs with 2 paraphrases",
             pct(sum(r["judge_scores"].get("paraphrases_equivalent") is True for r in v1p), len(v1p)),
             pct(sum(len(r["paraphrases"]) == 2 for r in v2p), len(v2p)))
        line("application share (generated)", pct(sum(r["q_type"] == "application" for r in v1), len(v1)),
             pct(sum(r["q_type"] == "application" for r in v2s), len(v2s)))
        line("hard share (generated)", pct(sum(r["difficulty"] == "hard" for r in v1), len(v1)),
             pct(sum(r["difficulty"] == "hard" for r in v2s), len(v2s)))
        extra = [r for r in rows if r["chunk_id"] not in same]
        if extra:
            rep(f"\nnew targeted chunks: {pct(sum(r['passed_filters'] for r in extra), len(extra))} passed; "
                f"passed by dimension: {dict(Counter(r['dimension'] for r in extra if r['passed_filters']))}")

    usage = [json.loads(line) for line in repo_path(cfg["llm"]["usage_log"]).open(encoding="utf-8")]
    rep("\n-- Cost (from logs/llm_usage.jsonl; cached re-runs cost $0) --")
    total = 0.0
    for st in (f"qa_extract:{args.run}", f"qa_generate:{args.run}", f"qa_filter:{args.run}"):
        for role in ("generator", "judge"):
            us = [u for u in usage if u["stage"] == st and u["role"] == role and not u["cached"]]
            if not us:
                continue
            c = sum(u["cost"] for u in us)
            total += c
            rep(f"  {st:24s} {role:9s} calls={len(us):4d} in={sum(u['input'] for u in us):>9,} "
                f"out={sum(u['output'] for u in us):>9,} ${c:.4f}")
    rep(f"  TOTAL ${total:.4f}  (${total / max(n, 1):.4f} per generated pair, "
        f"${total / max(len(passed), 1):.4f} per passed pair)")
    v1c = sum(u["cost"] for u in usage if u["stage"] in ("qa_extract", "qa_generate", "qa_filter") and not u["cached"])
    rep(f"  v1 total for comparison: ${v1c:.4f} for 100 pairs")

    rng = random.Random(args.seed)
    def show(r):
        js = r["judge_scores"]
        tag = " | REPAIRED" if r["repair"] else ""
        rep(f"\n[{r['qa_id']}] {r['q_type']} | {r['difficulty']} | {r['dimension']} | "
            f"grounding={js.get('grounding')} standalone={js.get('standalone')}{tag}"
            + (f" | REJECTED: {r['reject_reason']}" if r["reject_reason"] else ""))
        if r["repair"]:
            rep(f"  before: Q: {r['repair']['original_question']}")
            rep(f"  repair reason: {r['repair']['reason'][:300]}")
        rep(f"Q: {r['question']}")
        for x in r["paraphrases"]:
            rep(f"   ~ {x}")
        rep(f"A: {r['answer']}")
        rep(f"citation: {r['citation']}")
        if r["reject_reason"]:
            rep(f"judge: {js.get('grounding_rationale', '')} | {js.get('standalone_rationale', '')}")
    rejected = [r for r in rows if not r["passed_filters"]]
    rep_ok = [r for r in passed if r["repair"]]
    plain = [r for r in passed if not r["repair"]]
    sample_pass = rng.sample(rep_ok, min(2, len(rep_ok))) + rng.sample(plain, min(5 - min(2, len(rep_ok)), len(plain)))
    rep("\n-- 5 random passed examples (incl. repaired) --")
    for r in sample_pass:
        show(r)
    rep("\n-- 5 random rejected examples --")
    for r in rng.sample(rejected, min(5, len(rejected))):
        show(r)
    rep("\nthis run (stage 3):")
    for ln in llm.summary():
        rep(ln)
    rep.save()


if __name__ == "__main__":
    main()
