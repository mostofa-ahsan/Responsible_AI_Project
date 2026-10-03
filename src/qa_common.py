"""Shared helpers for the Phase 4 QA stages (qa_extract, qa_generate, qa_filter)."""

import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from chunk import read_jsonl
from utils import repo_path


def ws(text):
    return re.sub(r"\s+", " ", text).strip()


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_chunks(cfg):
    return {c["chunk_id"]: c for c in read_jsonl(repo_path(cfg["paths"]["chunks"]) / "chunks.jsonl")}


def load_metadata(cfg):
    with (repo_path(cfg["paths"]["parsed"]) / "metadata.csv").open(encoding="utf-8") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def flagged_docs(cfg):
    path = repo_path(cfg["paths"]["parsed"]) / "metadata_review.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {r["doc_id"] for r in csv.DictReader(f)}


def citation_title(meta_row):
    """Publisher metadata often uses ';' for the subtitle separator."""
    return ws(meta_row.get("title", "")).replace("; ", ": ")


def format_citation(title, pages):
    pages = sorted({p for p in pages if p})
    if not pages:
        return f"({title})"
    if len(pages) == 1 or pages[0] == pages[-1]:
        return f"({title}, p. {pages[0]})"
    return f"({title}, pp. {pages[0]}-{pages[-1]})"


class PageLookup:
    """Find the page of an evidence span via the parsed blocks a chunk was built from."""

    def __init__(self, cfg):
        self.dir = repo_path(cfg["paths"]["parsed"])
        self._docs = {}

    def blocks(self, doc_id):
        if doc_id not in self._docs:
            p = json.loads((self.dir / f"{doc_id}.json").read_text(encoding="utf-8"))
            self._docs[doc_id] = {b["block_id"]: b for b in p["blocks"]}
        return self._docs[doc_id]

    def page_of(self, chunk, evidence):
        blocks = self.blocks(chunk["doc_id"])
        ev = ws(evidence)
        probe = ev[:80]
        for bid in chunk["block_ids"]:
            b = blocks.get(bid)
            if b and (ev in ws(b["text"]) or probe in ws(b["text"])):
                return b["page"]
        return chunk["page_start"]


def select_pilot(cfg, chunks, log):
    """Deterministic pilot sample: n_docs (n_books books, rest articles), chunks_per_doc each."""
    pc = cfg["qa"]["pilot"]
    rng = random.Random(pc["seed"])
    flagged = flagged_docs(cfg) if pc["exclude_flagged_metadata"] else set()
    by_doc = {}
    for c in chunks.values():
        if c["mine"] and c["chunk_type"] == "text":
            by_doc.setdefault(c["doc_id"], []).append(c)
    cands = {f: sorted(d for d, cs in by_doc.items()
                       if len(cs) >= pc["min_mine_chunks"] and d not in flagged
                       and cs[0]["folder"] == f)
             for f in ("book", "article")}
    docs = rng.sample(cands["book"], pc["n_books"]) + \
        rng.sample(cands["article"], pc["n_docs"] - pc["n_books"])
    picked = []
    for d in docs:
        cs = sorted(by_doc[d], key=lambda c: c["chunk_id"])
        picked += sorted(rng.sample(cs, pc["chunks_per_doc"]), key=lambda c: c["chunk_id"])
    log.info(f"pilot docs (excluding {len(flagged)} metadata-flagged): {docs}")
    return docs, [c["chunk_id"] for c in picked]


def run_parallel(fn, items, max_workers, log):
    """Run fn(item) concurrently; yields (item, result | Exception)."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                yield it, fut.result()
            except Exception as e:  # noqa: BLE001 - logged and reported per item
                log.error(f"{it}: {type(e).__name__}: {e}")
                yield it, e
