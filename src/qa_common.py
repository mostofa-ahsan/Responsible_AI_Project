"""Shared helpers for the Phase 4 QA stages (qa_extract, qa_generate, qa_filter)."""

import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from chunk import is_appendix_or_table, read_jsonl  # noqa: F401 (re-exported)
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


# --- v2 additions -----------------------------------------------------------
def banned_regex(cfg):
    return re.compile(r"\b(?:" + "|".join(f"(?:{p})" for p in cfg["qa"]["banned_phrases"]) + r")", re.I)


def run_paths(cfg, run):
    d = repo_path(cfg["paths"]["qa_pairs"])
    return {k: d / f"{run}{suffix}" for k, suffix in (
        ("chunks", "_chunks.json"), ("units", "_units.jsonl"), ("generated", "_generated.jsonl"),
        ("final", ".jsonl"), ("review", "_review.csv"))}


def select_run(cfg, run, chunks, log):
    """Chunk selection for a run; written once to <run>_chunks.json and reused."""
    paths = run_paths(cfg, run)
    if paths["chunks"].exists():
        return json.loads(paths["chunks"].read_text())
    rcfg = cfg["qa"].get("runs", {}).get(run, {})
    if run == "pilot":
        docs, ids = select_pilot(cfg, chunks, log)
        sel = {"docs": docs, "chunk_ids": ids, "origin": {i: "pilot" for i in ids}}
    else:
        base = select_run(cfg, rcfg["reuse_chunks_from"], chunks, log)
        ids = list(base["chunk_ids"])
        origin = {i: base.get("origin", {}).get(i, rcfg["reuse_chunks_from"]) for i in ids}
        ex = rcfg.get("extra_chunks")
        if ex:
            for cid, dim in select_extra(cfg, chunks, set(base["docs"]), set(ids), ex, log):
                ids.append(cid)
                origin[cid] = f"extra:{dim}"
        docs = sorted({chunks[i]["doc_id"] for i in ids}, key=lambda d: ids.index(
            next(i for i in ids if chunks[i]["doc_id"] == d)))
        sel = {"docs": docs, "chunk_ids": ids, "origin": origin}
    paths["chunks"].parent.mkdir(parents=True, exist_ok=True)
    paths["chunks"].write_text(json.dumps(sel, indent=2))
    return sel


def select_extra(cfg, chunks, used_docs, used_ids, ex, log):
    """Round-robin over target dimensions: the strongest mineable text chunk per dimension from a
    new, metadata-unflagged doc (one chunk per doc), alternating books and articles."""
    from chunk import Scorer
    dims_rx = Scorer(cfg).dims
    flagged = flagged_docs(cfg)
    rng = random.Random(ex["seed"])
    cands = {}
    for d in ex["dimensions"]:
        rows = []
        for c in chunks.values():
            if (c["mine"] and c["chunk_type"] == "text" and c["chunk_id"] not in used_ids
                    and c["doc_id"] not in used_docs and c["doc_id"] not in flagged
                    and d in c["dimensions_preview"] and not is_appendix_or_table(c)
                    and c["topic_score"] >= ex.get("min_topic_score", 5)
                    and not METHODS_SECTION_RX.search(" > ".join(c["section_path"]))):
                n_hits = len(dims_rx[d].findall(c["text"]))
                if n_hits < ex.get("min_dim_hits", 3):
                    continue
                per100 = n_hits / max(len(c["text"].split()), 1) * 100
                rows.append((per100 + 0.1 * c["density"] + rng.random() * 1e-6, c))
        cands[d] = [c for _, c in sorted(rows, key=lambda x: x[0], reverse=True)]
    picked, docs, folder_turn = [], set(), ["article", "book"]
    i = 0
    while len(picked) < ex["n"] and any(cands.values()):
        d = ex["dimensions"][i % len(ex["dimensions"])]
        want = folder_turn[len(picked) % 2]
        pool = [c for c in cands[d] if c["doc_id"] not in docs]
        c = next((c for c in pool if c["folder"] == want), pool[0] if pool else None)
        if c:
            picked.append((c["chunk_id"], d))
            docs.add(c["doc_id"])
            cands[d].remove(c)
        else:
            cands[d] = []
        i += 1
    log.info(f"extra chunks: {picked}")
    return picked


METHODS_SECTION_RX = re.compile(
    r"method|research design|data collection|data analysis|participants|sampl|procedure|instrument|"
    r"search strategy|inclusion|screening|prisma|measures|analytical approach|limitations", re.I)

HUMAN_COLS = ("human_verdict", "human_notes")


def write_review_csv(path, header, rows, key="qa_id"):
    """Write a review CSV, carrying over filled human_* columns from an existing file by key."""
    old = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if any(r.get(h, "").strip() for h in HUMAN_COLS):
                    old[r[key]] = {h: r.get(h, "") for h in HUMAN_COLS}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(header) + list(HUMAN_COLS))
        w.writeheader()
        for r in rows:
            w.writerow({**{h: "" for h in HUMAN_COLS}, **r, **old.get(r[key], {})})
    return len(old)
