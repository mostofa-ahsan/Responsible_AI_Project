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
import sys
import json
import random
import re
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from llm import LLM, BillingError, Unavailable
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
undefined acronyms, references to unseen context, or wording that presupposes a source ("is
described as", "according to the ...", "the following"). A question may name a specific study,
book or framework by its content.

value_score (1-5): is the pair about responsible AI / AI in higher education (or the institutional
context it depends on) and worth learning?
  5 = core domain knowledge: a concept, finding, framework, risk or recommendation people should know
  4 = useful domain knowledge, somewhat specific
  3 = acceptable but narrow
  2 = low value: research methodology, search or inclusion criteria, which regions/countries/samples
      a study covered, course logistics or assignment weights, or a single table row
  1 = trivia: bibliographic details, signposting, or unrelated to the domain
A circular answer, one that only restates or rephrases the question without adding information
(e.g. Q "What does it mean that X was designed to be modular?" A "X was designed to be modular"),
scores value at most 2 regardless of topic.

question_well_formed: true if the question is grammatical, coherent and reads naturally. False if
it is garbled, ungrammatical, truncated, self-answering, or contrived (e.g. "What should a 2025
scoping review ... say about ...").

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
    value_score: int = Field(ge=1, le=5)
    value_rationale: str = Field(description="One short sentence")
    question_well_formed: bool
    paraphrase_equivalent: list[bool] = Field(description="One boolean per paraphrase, in order")


class ParaCheck(BaseModel):
    equivalent: list[bool] = Field(description="One boolean per candidate, in order")


SENT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def sentences(text):
    return len([s for s in SENT_RX.split(text.strip()) if s.strip()])


def evidence_sentences(evidence):
    return sum(sentences(span) for span in evidence.split(" … ") if span.strip())


class Checker:
    def __init__(self, cfg, llm, judge_role="judge"):
        self.cfg, self.llm, self.judge_role = cfg, llm, judge_role
        self.banned = banned_regex(cfg)
        self.meta = re.compile("|".join(f"(?:{x})" for x in cfg["qa"].get("meta_text_patterns", [])), re.I) \
            if cfg["qa"].get("meta_text_patterns") else None
        self.qcfg = cfg["qa"]

    def check(self, p):
        """Pre-filter, judge and format-check one pair. Returns (fails, scores)."""
        fails, scores = [], {}
        m = self.banned.search(p["question"])
        scores["prefilter_question"] = m.group(0) if m else None
        scores["prefilter_paraphrases"] = [bool(self.banned.search(x)) for x in p["paraphrases"]]
        meta = None
        if self.meta:
            for field in [p["question"], p["answer"]] + list(p["paraphrases"]):
                mm = self.meta.search(field)
                if mm:
                    meta = mm.group(0)
                    break
        scores["meta_text"] = meta
        if meta:
            fails.append(f"meta_text('{meta}')")
        if m:
            fails.append(f"not_standalone(prefilter:'{m.group(0)}')")
        else:
            paras = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(p["paraphrases"])) or "(none)"
            j = self.llm.complete(PROMPT.format(question=p["question"], paraphrases=paras,
                                                answer=p["answer"], evidence=p["evidence"]),
                                  SYSTEM, role=self.judge_role, json_schema=Judgement)
            jp = j.parsed
            eq = (jp.paraphrase_equivalent + [False] * len(p["paraphrases"]))[:len(p["paraphrases"])]
            scores.update(grounding=jp.grounding_score, grounding_rationale=jp.grounding_rationale,
                          standalone=jp.standalone, standalone_rationale=jp.standalone_rationale,
                          value=jp.value_score, value_rationale=jp.value_rationale,
                          well_formed=jp.question_well_formed,
                          paraphrase_equivalent=eq, judge_model=j.served_model)
            if jp.grounding_score < self.qcfg["min_grounding"]:
                fails.append(f"grounding({jp.grounding_score})")
            if not jp.standalone:
                fails.append("not_standalone(judge)")
            if jp.value_score < self.qcfg.get("min_value", 0):
                fails.append(f"low_value({jp.value_score})")
            if not jp.question_well_formed:
                fails.append("garbled_question")
        lo, hi = self.qcfg["sentence_range"][p["q_type"]]
        if evidence_sentences(p["evidence"]) <= 1:
            lo, hi = 1, max(hi, 5)     # single-sentence evidence: 1-5 sentences for every type
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
        try:
            new = regenerate_paraphrases(self.llm, self.cfg, p["question"], bad, len(bad)).parsed.paraphrases
        except Unavailable:                 # budget cap reached: drop the bad paraphrases
            info.update(dropped=len(bad), skipped_budget=True)
            return good, info
        new = [x.strip() for x in new if x.strip() and not self.banned.search(x)][:len(bad)]
        if new:
            try:
                chk = self.llm.complete(PARA_PROMPT.format(question=p["question"], cands="\n".join(
                    f"{i + 1}. {x}" for i, x in enumerate(new))), PARA_SYSTEM, role="judge",
                    json_schema=ParaCheck).parsed.equivalent
            except Unavailable:
                chk = []                    # unchecked paraphrases are not kept
            kept = [x for x, ok in zip(new, chk + [False] * len(new)) if ok]
        else:
            kept = []
        info["regenerated"] = len(kept)
        info["dropped"] = len(bad) - len(kept)
        return good + kept, info


def dedup(rows, cfg, log):
    """Drop a passing pair if its question is near-identical to an earlier kept pair anywhere in
    the run, or its answer is near-identical to an earlier kept pair from the same document.
    Vectors are L2-normalized, so a dot product is the cosine."""
    import numpy as np
    from embed import encode, load_model
    q_th, a_th = cfg["qa"]["dedup_cosine"], cfg["qa"].get("dedup_answer_cosine")
    idx = [i for i, r in enumerate(rows) if r["passed_filters"]]
    if not idx:
        return {"question": 0, "answer": 0}
    model = load_model(cfg["embed"])
    qv = encode(model, [rows[i]["question"] for i in idx], cfg["embed"]["batch_size"])
    av = encode(model, [rows[i]["answer"] for i in idx], cfg["embed"]["batch_size"]) if a_th else None
    kept_q = np.zeros_like(qv)
    n_kept = 0
    kept_ids = []
    kept_by_doc = defaultdict(list)            # doc_id -> positions j of kept pairs
    removed = {"question": 0, "answer": 0}
    for j, i in enumerate(idx):
        r = rows[i]
        bq, bq_id = 0.0, None
        if n_kept:
            sims = kept_q[:n_kept] @ qv[j]
            k = int(sims.argmax())
            bq, bq_id = float(sims[k]), kept_ids[k]
        ba, ba_id = 0.0, None
        if a_th and kept_by_doc[r["doc_id"]]:
            pos = kept_by_doc[r["doc_id"]]
            sims = av[pos] @ av[j]
            k = int(sims.argmax())
            ba, ba_id = float(sims[k]), rows[idx[pos[k]]]["qa_id"]
        r["judge_scores"]["max_question_cosine"] = round(bq, 3)
        r["judge_scores"]["max_answer_cosine_same_doc"] = round(ba, 3)
        if bq > q_th:
            r["passed_filters"], r["reject_reason"] = False, f"duplicate(question cos={bq:.3f} of {bq_id})"
            removed["question"] += 1
        elif a_th and ba > a_th:
            r["passed_filters"], r["reject_reason"] = False, f"duplicate(answer cos={ba:.3f} of {ba_id})"
            removed["answer"] += 1
        else:
            kept_q[n_kept] = qv[j]
            n_kept += 1
            kept_ids.append(r["qa_id"])
            kept_by_doc[r["doc_id"]].append(j)
    log.info(f"dedup removed: {removed}")
    return removed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="pilot_v2")
    ap.add_argument("--limit", type=int, help="only process the first N pairs")
    ap.add_argument("--skip-dedup", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", help="earlier run to compare with (default: the run's reuse_chunks_from)")
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
        repairable = [f for f in fails if f.startswith(("grounding", "not_standalone", "meta_text", "garbled_question"))]
        if any(f.startswith("low_value") for f in fails):
            repairable = []          # rewording cannot make a low-value pair worth learning
        if repairable and cfg["qa"]["repair_rounds"] > 0:
            reason = "; ".join(repairable) + ". " + " ".join(
                x for x in (scores.get("grounding_rationale") if any(f.startswith("grounding") for f in repairable) else None,
                            scores.get("standalone_rationale") if any(f.startswith("not_standalone") for f in repairable) else None,
                            "Remove meta-text that refers to the evidence or quoted text." if scores.get("meta_text") else None,
                            "Rewrite the question so it is grammatical and natural." if "garbled_question" in repairable else None)
                if x)
            try:
                fixed = repair_pair(llm, cfg, g, units[g["chunk_id"]], reason)
            except Unavailable:             # budget cap reached: keep the first-pass rejection
                rec.update(final_fails=fails + ["repair_skipped_budget"], final_scores=scores)
                return rec
            c = chunks[g["chunk_id"]]
            new = to_row(fixed.parsed, g["chunk_id"], int(g["qa_id"].rsplit("q", 1)[1]), c,
                         units[g["chunk_id"]], citation_title(meta[c["doc_id"]]), None, fixed.served_model)
            new["requested"] = g["requested"]
            if not new["unit_ids"]:            # repair cited no known unit: keep original evidence
                new.update(evidence=g["evidence"], unit_ids=g["unit_ids"], citation=g["citation"])
            try:
                f2, s2 = checker.check(new)
            except Unavailable:
                rec.update(final_fails=fails + ["repair_skipped_budget"], final_scores=scores)
                return rec
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
    dedup_removed = None if args.skip_dedup else dedup(rows, cfg, log)

    with paths["final"].open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    header = ["qa_id", "origin", "passed_filters", "reject_reason", "repaired", "q_type", "dimension",
              "difficulty", "question", "paraphrase_1", "paraphrase_2", "answer", "citation", "evidence",
              "grounding", "standalone", "value", "judge_rationale"]
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
            "value": js.get("value"),
            "judge_rationale": " | ".join(x for x in (js.get("grounding_rationale"), js.get("standalone_rationale"),
                                                      f"value: {js['value_rationale']}" if js.get("value_rationale") else None,
                                                      f"prefilter: {js['prefilter_question']}" if js.get("prefilter_question") else None) if x),
        })
    kept_human = write_review_csv(paths["review"], header, review)
    log.info(f"wrote {paths['final'].name} and {paths['review'].name} (kept {kept_human} human verdicts)")
    report(rows, cfg, args, llm, dedup_removed, paths)


def pct(a, b):
    return f"{a}/{b} ({a / max(b, 1):.0%})" if b else "n/a"


def first(r):
    """First-pass view of a row from any run version (v1 rows have no first_pass)."""
    fp = r.get("first_pass")
    if fp:
        return fp["passed"], fp["reject_reason"], fp["judge_scores"]
    return r["passed_filters"], r["reject_reason"], r["judge_scores"]


def run_metrics(rows, units, status, cost):
    n = len(rows)
    passed = [r for r in rows if r["passed_filters"]]
    fps = [first(r) for r in rows]
    judged = [js for _, _, js in fps if js.get("grounding") is not None]
    valued = [js["value"] for js in judged if "value" in js]
    reps = [r for r in rows if r.get("repair")]
    m = {
        "chunks used for generation": f"{sum(s['generate'] for s in status.values())}/{len(status)}" if status
        else f"{len({r['chunk_id'] for r in rows})}",
        "valid knowledge units": pct(sum(u["evidence_valid"] for u in units), len(units)) if units else "n/a",
        "pairs generated": str(n),
        "passed first pass (no repair)": pct(sum(ok for ok, _, _ in fps), n),
        "passed (final)": pct(len(passed), n),
        "grounding >= 4 (first pass, judged)": pct(sum(js["grounding"] >= 4 for js in judged), len(judged)),
        "grounding = 5 (first pass, judged)": pct(sum(js["grounding"] == 5 for js in judged), len(judged)),
        "standalone (first pass)": pct(sum("not_standalone" not in rr for _, rr, _ in fps), n),
        "value >= 3 (first pass, judged)": pct(sum(v >= 3 for v in valued), len(valued)) if valued else "n/a",
        "meta-text caught (first pass)": str(sum(bool(js.get("meta_text")) for _, _, js in fps)),
        "format ok (first pass)": pct(sum("format(" not in rr for _, rr, _ in fps), n),
        "repair attempted / passed": f"{len(reps)} / {sum(r['repair']['passed_after_repair'] for r in reps)}",
        # v1 kept both paraphrases regardless, so count only the ones its judge called equivalent
        "passing pairs with 2 paraphrases": pct(sum(
            len(r["paraphrases"]) == 2 and (bool(r.get("first_pass"))
                                            or r["judge_scores"].get("paraphrases_equivalent", True) is True)
            for r in passed), len(passed)),
        "application share (generated)": pct(sum(r["q_type"] == "application" for r in rows), n),
        "hard share (generated)": pct(sum(r["difficulty"] == "hard" for r in rows), n),
        "cost": f"${cost:.2f}",
        "cost per passed pair": f"${cost / max(len(passed), 1):.4f}",
    }
    return m


def run_cost(usage, run):
    if run == "pilot":
        stages = ("qa_extract", "qa_generate", "qa_filter")
    else:
        stages = (f"qa_extract:{run}", f"qa_generate:{run}", f"qa_filter:{run}")
    return sum(u["cost"] for u in usage if u["stage"] in stages and not u["cached"])


def report(rows, cfg, args, llm, dedup_removed, paths):
    rep = Report(f"qa_{args.run}", cfg)
    n = len(rows)
    passed = [r for r in rows if r["passed_filters"]]
    units = read_jsonl(paths["units"])
    status_path = paths["chunks"].with_name(f"{args.run}_chunk_status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    usage = [json.loads(line) for line in repo_path(cfg["llm"]["usage_log"]).open(encoding="utf-8")]

    rep(f"=== Phase 4a {args.run}: {n} QA pairs from {len({r['chunk_id'] for r in rows})} chunks, "
        f"{len({r['doc_id'] for r in rows})} docs ===")
    rep("single-chunk types only; closed_book_correct and retrieval_rank pending (null)")
    skipped = {cid: s["reason"] for cid, s in status.items() if not s["generate"]}
    rep(f"chunks skipped before generation: {len(skipped)} {skipped}")
    rep(f"knowledge units: {len(units)}; invalid: not verbatim {sum(not u.get('evidence_verbatim', u['evidence_valid']) for u in units)}, "
        f"ends mid-sentence {sum(u.get('evidence_complete') is False for u in units)}")
    reps = [r for r in rows if r["repair"]]
    rep(f"passed (final): {pct(len(passed), n)}   first pass: {pct(sum(r['first_pass']['passed'] for r in rows), n)}   "
        f"repair: attempted {len(reps)}, passed after repair {pct(sum(r['repair']['passed_after_repair'] for r in reps), len(reps))}")

    fps = [r["first_pass"] for r in rows]
    judged = [f for f in fps if "grounding" in f["judge_scores"]]
    rep("\n-- First-pass filters --")
    rep(f"prefilter (banned phrase in question): {sum(bool(f['judge_scores'].get('prefilter_question')) for f in fps)} "
        f"caught before the judge: {[f['judge_scores']['prefilter_question'] for f in fps if f['judge_scores'].get('prefilter_question')]}")
    rep(f"meta-text: {sum(bool(f['judge_scores'].get('meta_text')) for f in fps)}")
    rep(f"grounding distribution {dict(sorted(Counter(f['judge_scores']['grounding'] for f in judged).items()))}")
    rep(f"value distribution {dict(sorted(Counter(f['judge_scores'].get('value') for f in judged).items(), key=lambda kv: (kv[0] is None, kv[0])))}")
    rep(f"standalone (prefilter + judge): {pct(sum('not_standalone' not in f['reject_reason'] for f in fps), n)}")
    rep(f"format: {pct(sum('format(' not in f['reject_reason'] for f in fps), n)}")
    pf = [r["paraphrase_fix"] for r in rows if r.get("paraphrase_fix")]
    rep(f"paraphrases: {sum(x['initial_bad'] for x in pf)} bad on passing pairs -> {sum(x['regenerated'] for x in pf)} "
        f"regenerated, {sum(x['dropped'] for x in pf)} dropped")
    if dedup_removed is not None:
        rep(f"dedup removed: {dedup_removed['question']} by question, {dedup_removed['answer']} by answer (same doc)")
    rep(f"final reject reasons: {dict(Counter(x.split('(')[0] for r in rows for x in r['reject_reason'].split(';') if x))}")
    for key in ("q_type", "difficulty", "dimension"):
        g, pp = Counter(r[key] for r in rows), Counter(r[key] for r in passed)
        rep(f"\n-- {key}: generated -> passed --")
        for k, v in g.most_common():
            rep(f"  {k:24s} {v:4d} ({v / n:.0%}) -> {pp[k]:4d}")

    cmp_run = args.compare or (cfg["qa"].get("runs", {}).get(args.run, {}).get("reuse_chunks_from"))
    if cmp_run and run_paths(cfg, cmp_run)["final"].exists():
        cp = run_paths(cfg, cmp_run)
        old_rows = read_jsonl(cp["final"])
        old_units = read_jsonl(cp["units"])
        osp = cp["chunks"].with_name(f"{cmp_run}_chunk_status.json")
        old_status = json.loads(osp.read_text()) if osp.exists() else {}
        same = set(json.loads(cp["chunks"].read_text())["chunk_ids"])
        new_rows = [r for r in rows if r["chunk_id"] in same]
        new_status = {k: v for k, v in status.items() if k in same}
        a = run_metrics(old_rows, old_units, old_status, run_cost(usage, cmp_run))
        b = run_metrics(new_rows, [u for u in units if u["chunk_id"] in same], new_status, run_cost(usage, args.run))
        rep(f"\n-- {cmp_run} vs {args.run} on the same {len(same)} chunks --")
        rep(f"{'':38s} {cmp_run:>16s} {args.run:>16s}")
        for k in a:
            rep(f"{k:38s} {a[k]:>16s} {b[k]:>16s}")

    rep("\n-- Cost (from logs/llm_usage.jsonl; cached calls cost $0) --")
    total = 0.0
    for st in (f"qa_extract:{args.run}", f"qa_generate:{args.run}", f"qa_filter:{args.run}"):
        for role in ("generator", "judge"):
            us = [u for u in usage if u["stage"] == st and u["role"] == role and not u["cached"]]
            if us:
                c = sum(u["cost"] for u in us)
                total += c
                rep(f"  {st:24s} {role:9s} calls={len(us):4d} in={sum(u['input'] for u in us):>9,} "
                    f"out={sum(u['output'] for u in us):>9,} ${c:.4f}")
    rep(f"  TOTAL ${total:.4f}  (${total / max(n, 1):.4f} per generated pair, ${total / max(len(passed), 1):.4f} per passed pair)")

    rng = random.Random(args.seed)

    def show(r):
        js = r["judge_scores"]
        tag = " | REPAIRED" if r["repair"] else ""
        rep(f"\n[{r['qa_id']}] {r['q_type']} | {r['difficulty']} | {r['dimension']} | grounding={js.get('grounding')} "
            f"standalone={js.get('standalone')} value={js.get('value')}{tag}"
            + (f" | REJECTED: {r['reject_reason']}" if r["reject_reason"] else ""))
        if r["repair"]:
            rep(f"  before: Q: {r['repair']['original_question']}")
            rep(f"  repair reason: {r['repair']['reason'][:250]}")
        rep(f"Q: {r['question']}")
        for x in r["paraphrases"]:
            rep(f"   ~ {x}")
        rep(f"A: {r['answer']}")
        rep(f"citation: {r['citation']['title']}, pages {r['citation']['pages']}")
        if r["reject_reason"]:
            rep(f"judge: {js.get('grounding_rationale', '')} | {js.get('standalone_rationale', '')} | {js.get('value_rationale', '')}")
    rejected = [r for r in rows if not r["passed_filters"]]
    rep_ok = [r for r in passed if r["repair"]]
    plain = [r for r in passed if not r["repair"]]
    sample_pass = rng.sample(rep_ok, min(2, len(rep_ok))) + rng.sample(plain, min(5 - min(2, len(rep_ok)), len(plain)))
    by_reason = defaultdict(list)
    for r in rejected:
        by_reason[r["reject_reason"].split("(")[0]].append(r)
    sample_rej = []
    while len(sample_rej) < min(5, len(rejected)):
        for k in sorted(by_reason):
            if by_reason[k] and len(sample_rej) < 5:
                sample_rej.append(by_reason[k].pop(rng.randrange(len(by_reason[k]))))
    rep("\n-- 5 random passed examples (incl. repaired) --")
    for r in sample_pass:
        show(r)
    rep("\n-- 5 rejected examples (spread over reasons) --")
    for r in sample_rej:
        show(r)
    rep.save()


if __name__ == "__main__":
    try:
        main()
    except BillingError as e:
        print(f"BILLING ERROR, stopping: {e}", file=sys.stderr)
        sys.exit(3)
