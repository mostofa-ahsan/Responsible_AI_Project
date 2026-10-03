"""Phase 4 stage 2: generate QA pairs from validated knowledge units (generator model).

Slots: each usable chunk gets questions_per_chunk slots. Types come from qa.type_mix
restricted to the run's enabled types, difficulties from qa.difficulty_mix (both by
largest remainder); hard slots go to application/explanation first and easy slots
to factual/definition first, then slots are shuffled with a fixed seed.

The generator sees only the chunk's valid knowledge units (not the raw chunk) and
must restate only what the evidence says. The answer is stored without a citation;
citation {title, pages} is a separate field built from the evidence pages.
Training formats decide later whether to append it (closed-book: no; RAG/RAFT: yes).

Also provides repair_pair() and regenerate_paraphrases(), used by qa_filter.

Writes data/qa_pairs/<run>_generated.jsonl. Resumable by chunk_id.

Usage:
    python src/qa_generate.py --run pilot_v2 [--limit N]
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from llm import LLM
from qa_common import (append_jsonl, citation_title, load_chunks, load_metadata, read_jsonl,
                       run_parallel, run_paths, select_run)
from utils import Report, get_logger, load_config

TYPE_GUIDE = {
    "factual": "asks for a specific fact, number, finding or named item stated in the evidence",
    "definition": "asks what a term, concept, or framework component means or consists of",
    "explanation": "asks why or how something happens or matters, where the evidence itself "
                   "states the reason or mechanism",
    "application": "poses a realistic higher-education scenario (a university, instructor or "
                   "student situation) and asks what the evidence says applies or should be done",
}
DIFFICULTY_GUIDE = ("easy = restates one unit; medium = combines two units or explains a stated "
                    "reason; hard = needs several units together or applies them to a scenario")

DIMENSION_GUIDE = """governance: AI policies, oversight, accountability, regulation, ethics committees
leadership_strategy: institutional strategy, vision, leadership, investment, change management
faculty_readiness: teacher/faculty competence, attitudes, training, professional development
student_ai_literacy: students' AI literacy, skills, critical evaluation, appropriate use
teaching_learning: pedagogy, curriculum, course design, personalized learning, tutoring
assessment: assessment design, grading, academic integrity, plagiarism, AI detection
privacy_security: data protection, privacy, security, consent, surveillance
equity_accessibility: equity, inclusion, bias, fairness, accessibility, digital divide
procurement_technology: tools, platforms, infrastructure, vendors, licensing, IT support
monitoring_improvement: evaluation, metrics, monitoring, audits, maturity, continuous improvement
general: responsible-AI knowledge that fits none of the above"""

BANNED_FOR_PROMPT = ['"the evidence"', '"the text"', '"the passage"', '"the author(s)"',
                     '"the authors above"', '"this chapter/section/study/paper/review/framework"',
                     '"the study/review/report" (without saying which)', '"the recommendations say"',
                     '"according to the source"', '"as mentioned above"', '"in this context"']


def system_prompt(cfg):
    rng = cfg["qa"]["sentence_range"]
    return f"""You write question-answer pairs for a dataset that will fine-tune and evaluate a \
model on responsible AI in higher education. Every pair must be grounded in the knowledge units \
you are given, which were extracted verbatim from a book or journal article.

Question types:
{chr(10).join(f"- {k}: {v}" for k, v in TYPE_GUIDE.items())}
Difficulty: {DIFFICULTY_GUIDE}.

Questions:
- Standalone: a reader who has never seen the source must understand exactly what is asked.
  Never use any of these phrases (or close variants): {", ".join(BANNED_FOR_PROMPT)}.
  When a fact belongs to a specific study, book or framework, identify it by its content
  (e.g. "a 2025 scoping review of ChatGPT use in nursing education", "the AIware competency
  model for K-12") rather than by a pronoun or "the study".
- One clear ask per question; no multi-part questions.
- Exactly 2 paraphrases that ask for exactly the same information as the question (same scope,
  same entities, same answer), in different words. Do not narrow, broaden or add conditions.
- If a requested type or difficulty cannot be supported by the units, use the closest one the
  units do support and report what you actually wrote in q_type and difficulty.

Answers:
- Restate ONLY what the evidence says. No inference, synthesis, interpretation, evaluation or
  added connective claims ("this shows that", "therefore", "which means") unless the evidence
  itself states them. Do not combine units into a conclusion the evidence does not state.
- Length: {rng['factual'][0]}-{rng['factual'][1]} sentences for factual and definition,
  {rng['explanation'][0]}-{rng['explanation'][1]} sentences for explanation and application.
- No outside knowledge. Do not add a citation; it is stored separately.

Also assign one readiness dimension:
{DIMENSION_GUIDE}"""


PROMPT = """Source: {title}
Section: {section}

Knowledge units (id, type, unit, verbatim evidence):
{units}

Write {n} question-answer pairs, one per slot, in this order:
{slots}
Use different units across pairs where possible, and do not ask the same thing twice."""

REPAIR_PROMPT = """A reviewer rejected this question-answer pair.

Reason: {reason}

Question: {question}
Paraphrases: {paraphrases}
Answer: {answer}
Intended type: {q_type}; difficulty: {difficulty}

Knowledge units the pair must be grounded in (id, verbatim evidence):
{units}

Rewrite the pair to fix the problem. Keep the same type and topic. The answer must restate only
what the evidence says; the question and paraphrases must be standalone and avoid the banned
phrases. You may use any of the listed units."""

PARAPHRASE_PROMPT = """Question: {question}

These paraphrases were rejected because they do not ask for exactly the same information
(or use a banned phrase):
{bad}

Write {n} new paraphrase(s) of the question: same scope, same entities, same answer, different
wording, standalone."""


class QA(BaseModel):
    q_type: Literal["factual", "definition", "explanation", "application"]
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    paraphrases: list[str] = Field(description="Exactly 2 paraphrases of the question")
    answer: str = Field(description="Restates only the cited units' evidence; no citation")
    unit_ids: list[str] = Field(description="ids of the units whose evidence supports the answer")
    dimension: Literal["governance", "leadership_strategy", "faculty_readiness", "student_ai_literacy",
                       "teaching_learning", "assessment", "privacy_security", "equity_accessibility",
                       "procurement_technology", "monitoring_improvement", "general"]


class QASet(BaseModel):
    pairs: list[QA]


class Paraphrases(BaseModel):
    paraphrases: list[str]


def largest_remainder(weights, total):
    raw = {k: total * w / sum(weights.values()) for k, w in weights.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:total - sum(counts.values())]:
        counts[k] += 1
    return counts


def allocate_slots(cfg, enabled, n_chunks, seed):
    """List of (type, difficulty) slots over all chunks."""
    total = n_chunks * cfg["qa"]["questions_per_chunk"]
    tcounts = largest_remainder({t: cfg["qa"]["type_mix"][t] for t in enabled}, total)
    dcounts = largest_remainder(cfg["qa"]["difficulty_mix"], total)
    rng = random.Random(seed)
    types = [t for t, n in tcounts.items() for _ in range(n)]
    rng.shuffle(types)
    # hard -> application, explanation first; easy -> factual, definition first
    hard_pref = {"application": 0, "explanation": 1, "definition": 2, "factual": 3}
    order = sorted(range(total), key=lambda i: (hard_pref[types[i]], rng.random()))
    diff = [None] * total
    for i in order[:dcounts["hard"]]:
        diff[i] = "hard"
    rest = order[dcounts["hard"]:]
    for i in sorted(rest, key=lambda i: (-hard_pref[types[i]], rng.random()))[:dcounts["easy"]]:
        diff[i] = "easy"
    slots = [(t, d or "medium") for t, d in zip(types, diff)]
    rng.shuffle(slots)
    return slots, tcounts, dcounts


def unit_lines(us, with_unit=True):
    return "\n".join(f"[{u['unit_id'].split(':')[-1]}] ({u['type']}) "
                     + (f"{u['unit']}\n    evidence: " if with_unit else "evidence: ")
                     + f"\"{u['evidence']}\"" for u in us)


def to_row(qa, cid, i, chunk, units_by_short, title, plan_slot, generator):
    used = [units_by_short[s.split(":")[-1].strip("[] ")] for s in qa.unit_ids
            if s.split(":")[-1].strip("[] ") in units_by_short]
    return {
        "qa_id": f"{cid}:q{i}", "group_id": f"g:{cid}:q{i}", "chunk_id": cid,
        "doc_id": chunk["doc_id"], "question": qa.question.strip(),
        "paraphrases": [p.strip() for p in qa.paraphrases],
        "answer": qa.answer.strip(),
        "citation": {"title": title, "pages": sorted({u["page"] for u in used if u["page"]})
                     or [chunk["page_start"]]},
        "evidence": " … ".join(u["evidence"] for u in used),
        "unit_ids": [u["unit_id"] for u in used],
        "section_path": chunk["section_path"],
        "q_type": qa.q_type, "difficulty": qa.difficulty, "dimension": qa.dimension,
        "requested": {"q_type": plan_slot[0], "difficulty": plan_slot[1]} if plan_slot else None,
        "generator": generator,
    }


def repair_pair(llm, cfg, pair, units_by_short, reason):
    us = list(units_by_short.values())
    prompt = REPAIR_PROMPT.format(reason=reason, question=pair["question"],
                                  paraphrases=json.dumps(pair["paraphrases"]), answer=pair["answer"],
                                  q_type=pair["q_type"], difficulty=pair["difficulty"],
                                  units=unit_lines(us, with_unit=False))
    return llm.complete(prompt, system_prompt(cfg), role="generator", json_schema=QA)


def regenerate_paraphrases(llm, cfg, question, bad, n):
    prompt = PARAPHRASE_PROMPT.format(question=question, bad="\n".join(f"- {b}" for b in bad), n=n)
    return llm.complete(prompt, system_prompt(cfg), role="generator", json_schema=Paraphrases)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="pilot_v2")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("qa_generate", cfg)
    paths = run_paths(cfg, args.run)
    k = cfg["qa"]["questions_per_chunk"]
    chunks, meta = load_chunks(cfg), load_metadata(cfg)
    sel = select_run(cfg, args.run, chunks, log)
    status = json.loads(paths["chunks"].with_name(f"{args.run}_chunk_status.json").read_text())
    enabled = cfg["qa"]["runs"][args.run]["q_types"] if args.run in cfg["qa"].get("runs", {}) \
        else cfg["qa"]["pilot"]["q_types"]
    units = defaultdict(list)
    for u in read_jsonl(paths["units"]):
        if u["evidence_valid"]:
            units[u["chunk_id"]].append(u)

    usable = [cid for cid in sel["chunk_ids"] if status[cid]["generate"]]
    slots, tcounts, dcounts = allocate_slots(cfg, enabled, len(usable), cfg["qa"]["pilot"]["seed"])
    plan = {cid: slots[i * k:(i + 1) * k] for i, cid in enumerate(usable)}
    ids = usable[:args.limit] if args.limit else usable
    done = {g["chunk_id"] for g in read_jsonl(paths["generated"])}
    todo = [cid for cid in ids if cid not in done]
    log.info(f"{len(usable)} usable chunks; types {tcounts}; difficulty {dcounts}; {len(todo)} to generate")

    llm = LLM(cfg, f"qa_generate:{args.run}")
    system = system_prompt(cfg)

    def work(cid):
        c = chunks[cid]
        slot_lines = "\n".join(f"{i + 1}. type={t}, difficulty={d}" for i, (t, d) in enumerate(plan[cid]))
        prompt = PROMPT.format(title=citation_title(meta[c["doc_id"]]), section=" > ".join(c["section_path"]),
                               units=unit_lines(units[cid]), n=k, slots=slot_lines)
        return llm.complete(prompt, system, role="generator", json_schema=QASet)

    failures = []
    for cid, res in run_parallel(work, todo, cfg["llm"]["max_workers"], log):
        if isinstance(res, Exception):
            failures.append((cid, str(res)))
            continue
        c = chunks[cid]
        by_short = {u["unit_id"].split(":")[-1]: u for u in units[cid]}
        title = citation_title(meta[c["doc_id"]])
        rows = [to_row(qa, cid, i, c, by_short, title, plan[cid][i] if i < len(plan[cid]) else None,
                       res.served_model) for i, qa in enumerate(res.parsed.pairs)]
        append_jsonl(paths["generated"], rows)
        log.info(f"{cid}: {len(rows)} pairs")

    gen = [g for g in read_jsonl(paths["generated"]) if g["chunk_id"] in set(ids)]
    rep = Report(f"qa_generate_{args.run}", cfg)
    rep(f"=== Phase 4 stage 2 ({args.run}): {len(gen)} pairs from {len({g['chunk_id'] for g in gen})} chunks ===")
    rep(f"requested types: {tcounts}; requested difficulty: {dcounts}")
    rep(f"generated q_type: {dict(Counter(g['q_type'] for g in gen))}")
    rep(f"generated difficulty: {dict(Counter(g['difficulty'] for g in gen))}")
    rep(f"type changed by generator: {sum(g['requested'] and g['q_type'] != g['requested']['q_type'] for g in gen)}; "
        f"difficulty changed: {sum(g['requested'] and g['difficulty'] != g['requested']['difficulty'] for g in gen)}")
    rep(f"failures: {len(failures)} {failures[:3]}")
    rep("cost (this run):")
    for line in llm.summary():
        rep(line)
    rep.save()


if __name__ == "__main__":
    main()
