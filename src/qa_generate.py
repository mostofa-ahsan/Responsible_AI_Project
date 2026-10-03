"""Phase 4 stage 2: generate QA pairs from validated knowledge units (generator model).

Each chunk gets questions_per_chunk slots; slot types are drawn from qa.type_mix
restricted to the enabled types (pilot: factual, definition, explanation,
application) by largest remainder, shuffled with the pilot seed. The generator
sees only the chunk's valid knowledge units (not the raw chunk), so answers can
only use verified evidence. The citation "(Title, p. X)" is appended by code from
the evidence pages, not written by the model.

Writes data/qa_pairs/<run>_generated.jsonl. Resumable by chunk_id.

Usage:
    python src/qa_generate.py --pilot [--limit N]
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from llm import LLM
from qa_common import (append_jsonl, citation_title, format_citation, load_chunks, load_metadata,
                       read_jsonl, run_parallel)
from utils import Report, get_logger, load_config, repo_path

TYPE_GUIDE = {
    "factual": "asks for a specific fact, number, finding or named item stated in the evidence",
    "definition": "asks what a term, concept or framework component means or comprises",
    "explanation": "asks why or how something happens or matters (reasons, mechanisms, implications)",
    "application": "poses a realistic higher-education scenario (e.g. a university, instructor or "
                   "student situation) and asks what the evidence implies should be done",
}

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

SYSTEM = f"""You write question-answer pairs for a dataset that will fine-tune and evaluate a \
model on responsible AI in higher education. Every pair must be grounded in the knowledge units \
you are given, which were extracted verbatim from a book or journal article.

Question types:
{chr(10).join(f"- {k}: {v}" for k, v in TYPE_GUIDE.items())}

Rules for questions:
- Standalone: a reader who has never seen the source must understand exactly what is asked.
  Never write "this study", "the chapter", "the author", "the passage", "the text", "above".
  When a fact belongs to a specific study, book or framework, identify it by its content
  (e.g. "a 2025 scoping review of ChatGPT use in nursing education", "the AIware competency
  model for K-12") rather than by a pronoun.
- One clear ask per question; no multi-part questions.
- Each question gets exactly 2 paraphrases that ask the same thing in different words and
  have the same answer.
- If a requested type cannot be supported by the units (e.g. no definition is available), write a
  factual question instead and set q_type to factual.

Rules for answers:
- 2 to 5 sentences, written as a direct answer to the question.
- Use ONLY information in the cited units' evidence. No outside knowledge, no speculation.
- Do not add a citation; it is appended automatically.

Also assign one readiness dimension and a difficulty:
{DIMENSION_GUIDE}
difficulty: easy (one fact restated), medium (combines units or explains), hard (requires
reasoning across units or applying them to a scenario)."""

PROMPT = """Source: {title}
Section: {section}

Knowledge units (id, type, unit, verbatim evidence):
{units}

Write {n} question-answer pairs with these types, in this order: {types}.
Use different units across pairs where possible, and do not ask the same thing twice."""


class QA(BaseModel):
    q_type: Literal["factual", "definition", "explanation", "application"]
    question: str
    paraphrases: list[str] = Field(description="Exactly 2 paraphrases of the question")
    answer: str = Field(description="2-5 sentences, only from the cited units' evidence, no citation")
    unit_ids: list[str] = Field(description="ids of the units whose evidence supports the answer")
    dimension: Literal["governance", "leadership_strategy", "faculty_readiness", "student_ai_literacy",
                       "teaching_learning", "assessment", "privacy_security", "equity_accessibility",
                       "procurement_technology", "monitoring_improvement", "general"]
    difficulty: Literal["easy", "medium", "hard"]


class QASet(BaseModel):
    pairs: list[QA]


def allocate_types(cfg, n_chunks, seed):
    """Largest-remainder allocation of the enabled types over all slots, shuffled."""
    enabled = cfg["qa"]["pilot"]["q_types"]
    mix = {t: cfg["qa"]["type_mix"][t] for t in enabled}
    total = n_chunks * cfg["qa"]["questions_per_chunk"]
    raw = {t: total * w / sum(mix.values()) for t, w in mix.items()}
    counts = {t: math.floor(v) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - counts[t], reverse=True)[:total - sum(counts.values())]:
        counts[t] += 1
    slots = [t for t, n in counts.items() for _ in range(n)]
    random.Random(seed).shuffle(slots)
    return slots, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if not args.pilot:
        ap.error("only --pilot is implemented (the full run is Phase 5)")

    cfg = load_config()
    log = get_logger("qa_generate", cfg)
    out_dir = repo_path(cfg["paths"]["qa_pairs"])
    run, k = "pilot", cfg["qa"]["questions_per_chunk"]
    sel = json.loads((out_dir / f"{run}_chunks.json").read_text())
    gen_path = out_dir / f"{run}_generated.jsonl"
    chunks, meta = load_chunks(cfg), load_metadata(cfg)
    units = defaultdict(list)
    for u in read_jsonl(out_dir / f"{run}_units.jsonl"):
        if u["evidence_valid"]:
            units[u["chunk_id"]].append(u)

    slots, counts = allocate_types(cfg, len(sel["chunk_ids"]), cfg["qa"]["pilot"]["seed"])
    plan = {cid: slots[i * k:(i + 1) * k] for i, cid in enumerate(sel["chunk_ids"])}
    ids = sel["chunk_ids"][:args.limit] if args.limit else sel["chunk_ids"]
    done = {g["chunk_id"] for g in read_jsonl(gen_path)}
    todo = [cid for cid in ids if cid not in done and units[cid]]
    log.info(f"type allocation over {len(slots)} slots: {counts}; {len(todo)} chunks to generate")

    llm = LLM(cfg, "qa_generate")

    def work(cid):
        c = chunks[cid]
        ulines = "\n".join(f"[{u['unit_id'].split(':')[-1]}] ({u['type']}) {u['unit']}\n"
                           f"    evidence: \"{u['evidence']}\"" for u in units[cid])
        prompt = PROMPT.format(title=citation_title(meta[c["doc_id"]]), section=" > ".join(c["section_path"]),
                               units=ulines, n=k, types=", ".join(plan[cid]))
        return llm.complete(prompt, SYSTEM, role="generator", json_schema=QASet)

    failures = []
    for cid, res in run_parallel(work, todo, cfg["llm"]["max_workers"], log):
        if isinstance(res, Exception):
            failures.append((cid, str(res)))
            continue
        c = chunks[cid]
        by_short = {u["unit_id"].split(":")[-1]: u for u in units[cid]}
        title = citation_title(meta[c["doc_id"]])
        rows = []
        for i, qa in enumerate(res.parsed.pairs):
            used = [by_short[s.split(":")[-1].strip("[] ")] for s in qa.unit_ids
                    if s.split(":")[-1].strip("[] ") in by_short]
            answer = qa.answer.strip()
            cite = format_citation(title, [u["page"] for u in used])
            rows.append({
                "qa_id": f"{cid}:q{i}", "group_id": f"g:{cid}:q{i}", "chunk_id": cid,
                "doc_id": c["doc_id"], "question": qa.question.strip(),
                "paraphrases": [p.strip() for p in qa.paraphrases],
                "answer": f"{answer} {cite}",
                "evidence": " … ".join(u["evidence"] for u in used),
                "unit_ids": [u["unit_id"] for u in used],
                "section_path": c["section_path"],
                "page": sorted({u["page"] for u in used if u["page"]}) or [c["page_start"]],
                "q_type": qa.q_type,
                "requested_type": plan[cid][i] if i < len(plan[cid]) else None,
                "dimension": qa.dimension, "difficulty": qa.difficulty,
                "generator": res.served_model,
            })
        append_jsonl(gen_path, rows)
        log.info(f"{cid}: {len(rows)} pairs")

    gen = [g for g in read_jsonl(gen_path) if g["chunk_id"] in set(ids)]
    rep = Report("qa_generate", cfg)
    rep(f"=== Phase 4 stage 2 ({run}): {len(gen)} pairs from {len({g['chunk_id'] for g in gen})} chunks ===")
    rep(f"requested type allocation: {counts}")
    rep(f"generated q_type: {dict(Counter(g['q_type'] for g in gen))}")
    swapped = sum(g["q_type"] != g["requested_type"] for g in gen)
    rep(f"type swapped by generator (unsupported request): {swapped}")
    rep(f"no valid unit_ids: {sum(not g['unit_ids'] for g in gen)}; "
        f"paraphrase count != 2: {sum(len(g['paraphrases']) != 2 for g in gen)}")
    rep(f"failures: {len(failures)} {failures[:3]}")
    rep("cost (this run):")
    for line in llm.summary():
        rep(line)
    rep.save()


if __name__ == "__main__":
    main()
