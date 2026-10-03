"""Phase 4 stage 1: extract atomic knowledge units from chunks (generator model).

For each selected chunk, the generator returns units
    {unit, type (definition|claim|finding|framework|recommendation|causal), evidence}
where evidence must be a verbatim span of the chunk. Units whose evidence is not
a substring of the chunk (after whitespace normalization only) are kept in the
output with evidence_valid=false and are not used downstream.

Writes data/qa_pairs/<run>_units.jsonl (one row per unit) and
data/qa_pairs/<run>_chunks.json (the chunk selection). Resumable by chunk_id.

Usage:
    python src/qa_extract.py --pilot [--limit N]
"""

import argparse
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from llm import LLM
from qa_common import (PageLookup, append_jsonl, load_chunks, read_jsonl, run_parallel,
                       select_pilot, ws)
from utils import Report, get_logger, load_config, repo_path

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
- evidence is copied VERBATIM from the passage: one contiguous span (a sentence or a few
  consecutive sentences) that fully supports the unit. Do not paraphrase, fix typos, merge
  non-adjacent sentences, or add ellipses.
- Prefer substantive units a student or university administrator would want to know. Skip
  bibliographic details, section signposting ("Section 3 describes ..."), figure references and
  generic filler.
- If the passage has fewer good units than the minimum, return only the good ones."""

PROMPT = """Document: {title}
Section: {section}

<passage>
{text}
</passage>

Extract between {lo} and {hi} knowledge units from the passage."""


class Unit(BaseModel):
    unit: str = Field(description="The knowledge, restated in 1-2 standalone sentences")
    type: Literal["definition", "claim", "finding", "framework", "recommendation", "causal"]
    evidence: str = Field(description="Verbatim contiguous span from the passage supporting the unit")


class Units(BaseModel):
    units: list[Unit]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", action="store_true", help="use the pilot chunk selection")
    ap.add_argument("--limit", type=int, help="only process the first N chunks")
    args = ap.parse_args()
    if not args.pilot:
        ap.error("only --pilot is implemented (the full run is Phase 5)")

    cfg = load_config()
    log = get_logger("qa_extract", cfg)
    out_dir = repo_path(cfg["paths"]["qa_pairs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run = "pilot"
    sel_path, units_path = out_dir / f"{run}_chunks.json", out_dir / f"{run}_units.jsonl"
    chunks = load_chunks(cfg)

    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
    else:
        docs, ids = select_pilot(cfg, chunks, log)
        sel = {"docs": docs, "chunk_ids": ids}
        sel_path.write_text(json.dumps(sel, indent=2))
    todo_ids = sel["chunk_ids"][:args.limit] if args.limit else sel["chunk_ids"]
    done = {u["chunk_id"] for u in read_jsonl(units_path)}
    todo = [cid for cid in todo_ids if cid not in done]
    log.info(f"{len(todo_ids)} chunks selected, {len(todo)} to extract")

    llm = LLM(cfg, "qa_extract")
    pages = PageLookup(cfg)
    lo, hi = cfg["qa"]["units_per_chunk"]

    def work(cid):
        c = chunks[cid]
        prompt = PROMPT.format(title=c["title"], section=" > ".join(c["section_path"]),
                               text=c["text"], lo=lo, hi=hi)
        return llm.complete(prompt, SYSTEM, role="generator", json_schema=Units)

    failures = []
    for cid, res in run_parallel(work, todo, cfg["llm"]["max_workers"], log):
        if isinstance(res, Exception):
            failures.append((cid, str(res)))
            continue
        c = chunks[cid]
        norm_chunk = ws(c["text"])
        rows = []
        for i, u in enumerate(res.parsed.units):
            valid = ws(u.evidence) in norm_chunk and len(ws(u.evidence)) >= 20
            rows.append({
                "unit_id": f"{cid}:u{i:02d}", "chunk_id": cid, "doc_id": c["doc_id"],
                "unit": u.unit, "type": u.type, "evidence": u.evidence,
                "evidence_valid": valid, "section_path": c["section_path"],
                "page": pages.page_of(c, u.evidence) if valid else None,
                "model": res.served_model,
            })
        append_jsonl(units_path, rows)
        log.info(f"{cid}: {len(rows)} units, {sum(r['evidence_valid'] for r in rows)} valid evidence")

    units = [u for u in read_jsonl(units_path) if u["chunk_id"] in set(todo_ids)]
    rep = Report("qa_extract", cfg)
    rep(f"=== Phase 4 stage 1 ({run}): knowledge units from {len({u['chunk_id'] for u in units})} chunks ===")
    rep(f"docs: {', '.join(sel['docs'])}")
    valid = [u for u in units if u["evidence_valid"]]
    rep(f"units: {len(units)}  valid evidence (verbatim substring): {len(valid)} "
        f"({len(valid) / max(len(units), 1):.0%})")
    rep(f"by type (valid): {dict(Counter(u['type'] for u in valid))}")
    rep(f"failures: {len(failures)} {failures[:3]}")
    rep("cost (this run):")
    for line in llm.summary():
        rep(line)
    bad = [u for u in units if not u["evidence_valid"]][:3]
    if bad:
        rep("\nexamples of rejected evidence:")
        for u in bad:
            rep(f"  {u['unit_id']}: {u['evidence'][:160]!r}")
    rep.save()


if __name__ == "__main__":
    main()
