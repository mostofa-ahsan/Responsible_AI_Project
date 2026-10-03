"""Phase 4 stage 1: extract atomic knowledge units from chunks (generator model).

For each selected chunk, the generator returns units
    {unit, type (definition|claim|finding|framework|recommendation|causal), evidence}
where evidence must be a verbatim span of the chunk. Units whose evidence is not
a substring of the chunk (after whitespace normalization only) are kept with
evidence_valid=false and are not used downstream. Review methodology (search
strategy, inclusion criteria, screening, framework mechanics) is skipped.

A unit is also invalid if its evidence ends mid-sentence (no final . ! ? before closing
quotes/brackets or a trailing reference marker). Appendix sections and table-row chunks are
skipped without an API call. A chunk is used for generation only if it has >= qa.min_units
valid units; see <run>_chunk_status.json.

Writes data/qa_pairs/<run>_units.jsonl and <run>_chunks.json. Resumable by chunk_id.

Usage:
    python src/qa_extract.py --run pilot_v2 [--limit N]
"""

import argparse
import json
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from llm import LLM
from qa_common import (PageLookup, append_jsonl, is_appendix_or_table, load_chunks, read_jsonl,
                       run_parallel, run_paths, select_run, ws)
from utils import Report, get_logger, load_config

SYSTEM = """You extract knowledge units from passages of books and journal articles about \
responsible AI in higher education, to build a source-grounded question-answering dataset.

A knowledge unit is one atomic, self-contained piece of knowledge stated in the passage:
- definition: what a term or concept means
- claim: an assertion or argument the author makes
- finding: an empirical result (study outcome, statistic, observed effect)
- framework: a model, framework, taxonomy, set of principles or dimensions, and its parts
- recommendation: what institutions, educators, students or policymakers should do
- causal: a cause-effect or mechanism relationship

Rules:
- Each unit restates the knowledge in one or two plain sentences that make sense on their own:
  name the subject explicitly (no "this study", "the authors", "it").
- evidence is copied VERBATIM from the passage: one contiguous span of COMPLETE sentences (a
  sentence or a few consecutive sentences, ending with the sentence's final punctuation) that
  fully supports the unit. Never stop mid-sentence. Do not paraphrase, fix typos, merge
  non-adjacent sentences, or add ellipses.
- Prefer substantive units a student or university administrator would want to know about AI
  in higher education: concepts, findings, frameworks and their content, recommendations,
  risks, causes and effects.
- SKIP research-process details: search strategy, databases searched, inclusion/exclusion
  criteria, screening and PRISMA counts, sample recruitment, survey or interview procedures,
  statistical procedures and fit indices, and the mechanics of a review framework (e.g. how
  its guiding questions are worded). Keep the substantive findings those methods produced.
- SKIP bibliographic details, author affiliations, section signposting ("Section 3
  describes ..."), figure/table references and generic filler.
- It is fine to return few or zero units if the passage has little substantive knowledge."""

PROMPT = """Document: {title}
Section: {section}

<passage>
{text}
</passage>

Extract up to {hi} knowledge units from the passage (aim for {lo} or more if the passage supports it)."""


class Unit(BaseModel):
    unit: str = Field(description="The knowledge, restated in 1-2 standalone sentences")
    type: Literal["definition", "claim", "finding", "framework", "recommendation", "causal"]
    evidence: str = Field(description="Verbatim contiguous span from the passage supporting the unit")


class Units(BaseModel):
    units: list[Unit]


SENTENCE_END_RX = re.compile(r"[.!?][\"”’')\]]*(?:\s*\[\d+(?:\s*[,–-]\s*\d+)*\])?$")


def complete_sentence(evidence):
    return bool(SENTENCE_END_RX.search(ws(evidence)))


def chunk_status(cfg, chunks, sel, units):
    """Which chunks have enough valid units to generate from."""
    valid = Counter(u["chunk_id"] for u in units if u["evidence_valid"])
    status = {}
    for cid in sel["chunk_ids"]:
        c = chunks[cid]
        if is_appendix_or_table(c):
            status[cid] = {"valid_units": 0, "appendix_or_table": True, "generate": False,
                           "reason": "appendix_or_table"}
            continue
        need = cfg["qa"]["min_units"]
        status[cid] = {"valid_units": valid[cid], "appendix_or_table": False,
                       "generate": valid[cid] >= need,
                       "reason": "" if valid[cid] >= need else f"thin({valid[cid]}<{need})"}
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="pilot_v2")
    ap.add_argument("--limit", type=int, help="only process the first N chunks")
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("qa_extract", cfg)
    paths = run_paths(cfg, args.run)
    chunks = load_chunks(cfg)
    sel = select_run(cfg, args.run, chunks, log)
    todo_ids = sel["chunk_ids"][:args.limit] if args.limit else sel["chunk_ids"]
    done = {u["chunk_id"] for u in read_jsonl(paths["units"])}
    done |= set(json.loads(paths["chunks"].with_name(f"{args.run}_empty.json").read_text())
                if paths["chunks"].with_name(f"{args.run}_empty.json").exists() else [])
    todo = [cid for cid in todo_ids if cid not in done and not is_appendix_or_table(chunks[cid])]
    log.info(f"{len(todo_ids)} chunks selected, {len(todo)} to extract")

    llm = LLM(cfg, f"qa_extract:{args.run}")
    pages = PageLookup(cfg)
    lo, hi = cfg["qa"]["units_per_chunk"]

    def work(cid):
        c = chunks[cid]
        prompt = PROMPT.format(title=c["title"], section=" > ".join(c["section_path"]),
                               text=c["text"], lo=lo, hi=hi)
        return llm.complete(prompt, SYSTEM, role="generator", json_schema=Units)

    failures, empty = [], []
    for cid, res in run_parallel(work, todo, cfg["llm"]["max_workers"], log):
        if isinstance(res, Exception):
            failures.append((cid, str(res)))
            continue
        c = chunks[cid]
        norm_chunk = ws(c["text"])
        rows = []
        for i, u in enumerate(res.parsed.units):
            verbatim = ws(u.evidence) in norm_chunk and len(ws(u.evidence)) >= 20
            complete = complete_sentence(u.evidence)
            valid = verbatim and complete
            rows.append({
                "unit_id": f"{cid}:u{i:02d}", "chunk_id": cid, "doc_id": c["doc_id"],
                "unit": u.unit, "type": u.type, "evidence": u.evidence,
                "evidence_valid": valid, "evidence_verbatim": verbatim, "evidence_complete": complete,
                "section_path": c["section_path"],
                "page": pages.page_of(c, u.evidence) if valid else None,
                "model": res.served_model,
            })
        if rows:
            append_jsonl(paths["units"], rows)
        else:
            empty.append(cid)
        log.info(f"{cid}: {len(rows)} units, {sum(r['evidence_valid'] for r in rows)} valid evidence")
    if empty:   # remember chunks that legitimately produced no units, so resume skips them
        ep = paths["chunks"].with_name(f"{args.run}_empty.json")
        prev = json.loads(ep.read_text()) if ep.exists() else []
        ep.write_text(json.dumps(sorted(set(prev) | set(empty))))

    units = [u for u in read_jsonl(paths["units"]) if u["chunk_id"] in set(todo_ids)]
    status = chunk_status(cfg, chunks, sel, units)
    paths["chunks"].with_name(f"{args.run}_chunk_status.json").write_text(json.dumps(status, indent=2))

    rep = Report(f"qa_extract_{args.run}", cfg)
    rep(f"=== Phase 4 stage 1 ({args.run}): knowledge units from {len(todo_ids)} chunks ===")
    valid = [u for u in units if u["evidence_valid"]]
    rep(f"units: {len(units)}  valid evidence: {len(valid)} ({len(valid) / max(len(units), 1):.0%}); "
        f"not verbatim: {sum(not u.get('evidence_verbatim', u['evidence_valid']) for u in units)}, "
        f"ends mid-sentence: {sum(u.get('evidence_complete') is False for u in units)}")
    rep(f"by type (valid): {dict(Counter(u['type'] for u in valid))}")
    skipped = {cid: s for cid, s in status.items() if not s["generate"] and cid in set(todo_ids)}
    rep(f"chunks usable for generation: {len(todo_ids) - len(skipped)}/{len(todo_ids)}; "
        f"skipped as thin: {[(cid, s['reason']) for cid, s in skipped.items()]}")
    rep(f"failures: {len(failures)} {failures[:3]}")
    rep("cost (this run):")
    for line in llm.summary():
        rep(line)
    rep.save()


if __name__ == "__main__":
    main()
