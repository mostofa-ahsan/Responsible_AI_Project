"""Phase 3: structure-aware chunking, heuristic density scoring, mining selection.

Reads data/parsed/<doc_id>.json (+ metadata.csv) and writes data/chunks/chunks.jsonl:

    {chunk_id, doc_id, title, section_path, page_start, page_end, text, n_tokens,
     chunk_type (text|table), block_ids, overlap_tokens, split_paragraph, folder,
     doc_group, density, density_signals, topic_score, dimensions_preview, mine,
     mine_reason}

Chunks never cross a top-level section (section_path[0]) and never split a
paragraph (except a single paragraph longer than hard_max_tokens, which is split
at sentence boundaries and flagged). skip=true blocks are excluded. Token counts
use the base model's tokenizer (config base_model).

Resumable: docs already in chunks.jsonl are skipped. --rescore recomputes
density / mine / dimension preview for existing chunks without re-chunking.

Usage:
    python src/chunk.py [--limit N] [--doc DOC_ID ...] [--force] [--rescore]
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict

from utils import Report, get_logger, load_config, repo_path

SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


class Tok:
    def __init__(self, model):
        from transformers import AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(model)

    def count(self, texts):
        if not texts:
            return []
        return [len(ids) for ids in self.tk(texts, add_special_tokens=False)["input_ids"]]


# --- chunking ---------------------------------------------------------------
def doc_units(parsed):
    """Non-skip blocks as units, grouped by top-level section (in document order)."""
    sections, order = defaultdict(list), []
    prev_key = object()
    for b in parsed["blocks"]:
        if b["skip"]:
            continue
        key = b["section_path"][0] if b["section_path"] else ""
        # A top-level heading reappearing later (rare) starts a new group.
        if key != prev_key:
            order.append([])
            prev_key = key
        order[-1].append(b)
    return order


def tail_sentences(text, tok, budget):
    """Trailing whole sentences of text totalling <= budget tokens."""
    sents = SENT_SPLIT_RX.split(text)
    out = []
    for s in reversed(sents):
        cand = " ".join([s] + out)
        if tok.count([cand])[0] > budget:
            break
        out.insert(0, s)
    return " ".join(out)


def split_long_paragraph(b, tok, max_tokens):
    """Split one oversized paragraph at sentence boundaries into <= max_tokens pieces."""
    sents = SENT_SPLIT_RX.split(b["text"])
    counts = tok.count(sents)
    pieces, cur, cur_t = [], [], 0
    for s, n in zip(sents, counts):
        if cur and cur_t + n > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_t = [], 0
        cur.append(s)
        cur_t += n
    if cur:
        pieces.append(" ".join(cur))
    return [{**b, "text": p, "_split": True} for p in pieces]


def chunk_section(units, tok, ccfg):
    """Greedy packing of one top-level section's units into chunk specs."""
    mn, mx, hard = ccfg["min_tokens"], ccfg["max_tokens"], ccfg["hard_max_tokens"]
    expanded = []
    for u, n in zip(units, tok.count([u["text"] for u in units])):
        if u["type"] != "table" and n > hard:
            parts = split_long_paragraph(u, tok, mx)
            expanded += [(p, c) for p, c in zip(parts, tok.count([p["text"] for p in parts]))]
        else:
            expanded.append((u, n))

    specs = []          # each: {"units": [...], "overlap": str, "type": "text"|"table"}
    cur, cur_t, overlap = [], 0, ""

    def flush(with_overlap):
        nonlocal cur, cur_t, overlap
        if cur:
            specs.append({"units": cur, "overlap": overlap, "type": "text"})
            last_text = next((u["text"] for u in reversed(cur)
                              if u["type"] not in ("table", "heading")), "")
            overlap = tail_sentences(last_text, tok, ccfg["overlap_tokens"]) if with_overlap else ""
        cur, cur_t = [], 0

    for i, (u, n) in enumerate(expanded):
        if u["type"] == "table":
            if cur_t + n <= mx:
                cur.append(u)
                cur_t += n
                continue
            if n >= ccfg["table_alone_tokens"]:
                flush(with_overlap=True)
                # own chunk, with an adjacent caption if there is one
                cap = []
                if specs and specs[-1]["units"] and specs[-1]["units"][-1]["type"] == "caption":
                    cap = [specs[-1]["units"].pop()]
                    if not specs[-1]["units"]:
                        specs.pop()
                specs.append({"units": cap + [u], "overlap": "", "type": "table"})
                continue
            flush(with_overlap=False)
            cur, cur_t = [u], n
            continue

        if u["type"] == "heading" and cur_t >= mn:
            flush(with_overlap=False)       # clean break at a subsection
        elif cur and cur_t + n > mx and (cur_t >= mn or cur_t + n > hard):
            flush(with_overlap=True)
        cur.append(u)
        cur_t += n
    flush(with_overlap=False)

    # Fold a small trailing text chunk into the previous one if it fits.
    if len(specs) >= 2 and specs[-1]["type"] == specs[-2]["type"] == "text":
        tail, prev = specs[-1], specs[-2]
        t_tail = sum(tok.count([u["text"] for u in tail["units"]]))
        t_prev = sum(tok.count([u["text"] for u in prev["units"]]))
        if t_tail < mn / 2 and t_prev + t_tail <= hard:
            prev["units"] += tail["units"]
            specs.pop()
    # A table chunk that left an orphan caption-only spec behind is impossible (popped above);
    # drop specs that ended up with headings only.
    return [s for s in specs if any(u["type"] != "heading" for u in s["units"])]


def build_chunks(parsed, meta, tok, ccfg):
    chunks = []
    for units in doc_units(parsed):
        for spec in chunk_section(units, tok, ccfg):
            us = spec["units"]
            body = "\n\n".join(u["text"] for u in us)
            text = f"{spec['overlap']}\n\n{body}" if spec["overlap"] else body
            first = next((u for u in us if u["type"] != "heading"), us[0])
            pages = [p for u in us for p in (u["page"], u["page_end"]) if p]
            chunks.append({
                "chunk_id": f"{parsed['doc_id']}:{len(chunks):04d}",
                "doc_id": parsed["doc_id"],
                "title": meta.get("title", ""),
                "section_path": first["section_path"],
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "text": text,
                "n_tokens": 0,  # filled in batch below
                "chunk_type": spec["type"],
                "block_ids": [u["block_id"] for u in us],
                "overlap_tokens": tok.count([spec["overlap"]])[0] if spec["overlap"] else 0,
                "split_paragraph": any(u.get("_split") for u in us),
                "folder": parsed["folder"],
                "doc_group": parsed["doc_group"],
            })
    for c, n in zip(chunks, tok.count([c["text"] for c in chunks])):
        c["n_tokens"] = n
    return chunks


# --- scoring ----------------------------------------------------------------
class Scorer:
    def __init__(self, cfg):
        d = cfg["density"]
        self.scale, self.weights, self.caps = d["scale"], d["weights"], d.get("caps", {})
        self.signals = {k: re.compile(v) for k, v in d["signals"].items()}
        dc = cfg["dimensions"]
        self.min_hits = dc["min_hits"]
        self.dims = {name: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b", re.I)
                     for name, kws in dc["keywords"].items()}
        self.ccfg = cfg["chunk"]
        topic_terms = list(cfg["topic"]["keywords"]) + [k for kws in dc["keywords"].values() for k in kws]
        self.topic = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in sorted(set(topic_terms), key=len, reverse=True))
                                + r")", re.I)

    def score(self, c):
        text = c["text"]
        words = max(len(text.split()), 1)
        hits = {k: len(rx.findall(text)) for k, rx in self.signals.items()}
        per100 = words / 100
        raw = sum(self.weights[k] * min(v / per100, self.caps.get(k, float("inf"))) for k, v in hits.items())
        c["density"] = round(1 - math.exp(-raw / self.scale), 3)
        c["topic_score"] = round(len(self.topic.findall(text)) / per100, 2)
        c["density_signals"] = {k: v for k, v in hits.items() if v}
        c["dimensions_preview"] = [n for n, rx in self.dims.items()
                                   if len(rx.findall(text)) >= self.min_hits]
        if c["doc_id"] in self.ccfg["exclude_from_mining"]:
            c["mine"], c["mine_reason"] = False, "excluded_doc"
        elif c["n_tokens"] < self.ccfg["mine_min_tokens"]:
            c["mine"], c["mine_reason"] = False, "too_short"
        elif c["density"] < self.ccfg["density_threshold"]:
            c["mine"], c["mine_reason"] = False, "low_density"
        elif c["topic_score"] < self.ccfg.get("min_topic_score", 0):
            c["mine"], c["mine_reason"] = False, "off_topic"
        else:
            c["mine"], c["mine_reason"] = True, ""
        return c


# --- io ---------------------------------------------------------------------
def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, help="only chunk the first N (new) documents")
    ap.add_argument("--doc", nargs="+", help="only these doc_ids")
    ap.add_argument("--force", action="store_true", help="re-chunk everything")
    ap.add_argument("--rescore", action="store_true", help="only recompute density/mine/dimensions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    ccfg = cfg["chunk"]
    log = get_logger("chunk", cfg)
    parsed_dir = repo_path(cfg["paths"]["parsed"])
    out_dir = repo_path(cfg["paths"]["chunks"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "chunks.jsonl"
    scorer = Scorer(cfg)

    chunks = [] if args.force else read_jsonl(out)
    done = {c["doc_id"] for c in chunks}
    new_docs = 0
    if not args.rescore:
        with (parsed_dir / "metadata.csv").open(encoding="utf-8") as f:
            meta = {r["doc_id"]: r for r in csv.DictReader(f)}
        files = sorted(parsed_dir.glob("*.json"))
        todo = [f for f in files if f.stem not in done]
        if args.doc:
            todo = [f for f in todo if f.stem in set(args.doc)]
        if args.limit:
            todo = todo[:args.limit]
        if todo:
            log.info(f"Loading tokenizer {cfg['base_model']}")
            tok = Tok(cfg["base_model"])
        for i, f in enumerate(todo, 1):
            parsed = json.loads(f.read_text(encoding="utf-8"))
            doc_chunks = build_chunks(parsed, meta.get(parsed["doc_id"], {}), tok, ccfg)
            chunks += doc_chunks
            new_docs += 1
            log.info(f"[{i}/{len(todo)}] {parsed['doc_id']}: {len(doc_chunks)} chunks")
            if i % 10 == 0:   # checkpoint so a crash doesn't lose finished docs
                write_jsonl(out, [scorer.score(c) for c in chunks])

    chunks = [scorer.score(c) for c in chunks]
    write_jsonl(out, chunks)
    log.info(f"Wrote {len(chunks)} chunks ({new_docs} new docs) to {out}")
    report(chunks, cfg, args.seed, new_docs)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def report(chunks, cfg, seed, new_docs):
    ccfg = cfg["chunk"]
    rep = Report("chunk", cfg)
    toks = [c["n_tokens"] for c in chunks]
    rep(f"=== Phase 3 chunking: {len(chunks):,} chunks from {len({c['doc_id'] for c in chunks})} docs "
        f"({new_docs} chunked this run) ===")
    rep(f"tokenizer: {cfg['base_model']}   target {ccfg['min_tokens']}-{ccfg['max_tokens']} "
        f"(hard max {ccfg['hard_max_tokens']}), overlap ~{ccfg['overlap_tokens']}")
    rep(f"total tokens: {sum(toks):,}")
    rep(f"tokens per chunk: min {min(toks)}  median {int(statistics.median(toks))}  "
        f"p95 {pct(toks, 0.95)}  max {max(toks)}")
    for lo, hi in ((0, 150), (150, 500), (500, 1000), (1000, 1300), (1300, 10**9)):
        n = sum(lo <= t < hi for t in toks)
        rep(f"  {lo:>5}-{hi if hi < 10**9 else '':<5} {n:6,} ({n / len(toks):.0%})")

    per_doc = Counter(c["doc_id"] for c in chunks)
    folder = {c["doc_id"]: c["folder"] for c in chunks}
    rep("\n-- Chunks per doc --")
    for f in ("book", "article"):
        xs = [n for d, n in per_doc.items() if folder[d] == f]
        if xs:
            rep(f"{f:8s} docs={len(xs):3d}  total={sum(xs):,}  mean={statistics.mean(xs):.1f}  "
                f"median={statistics.median(xs):.0f}  min={min(xs)}  max={max(xs)}")

    table = [c for c in chunks if c["chunk_type"] == "table"]
    with_table = [c for c in chunks if c["chunk_type"] == "text" and "\n|" in c["text"]]
    rep(f"\ntable chunks (own chunk): {len(table)}; text chunks containing an attached table: {len(with_table)}")
    rep(f"paragraphs split at sentences (longer than hard max): "
        f"{sum(c['split_paragraph'] for c in chunks)} chunks")
    rep(f"chunks with overlap: {sum(c['overlap_tokens'] > 0 for c in chunks):,} "
        f"(mean {statistics.mean([c['overlap_tokens'] for c in chunks if c['overlap_tokens']] or [0]):.0f} tokens)")

    mine = [c for c in chunks if c["mine"]]
    reasons = Counter(c["mine_reason"] for c in chunks if not c["mine"])
    rep(f"\n-- Mining selection (density >= {ccfg['density_threshold']}, topic_score >= "
        f"{ccfg.get('min_topic_score', 0)}, >= {ccfg['mine_min_tokens']} tokens) --")
    rep(f"selected: {len(mine):,}/{len(chunks):,} chunks ({len(mine) / len(chunks):.0%}), "
        f"{sum(c['n_tokens'] for c in mine):,} tokens (Phase 5 generator input volume)")
    rep(f"not selected: {dict(reasons)}")
    for f in ("book", "article"):
        fc = [c for c in chunks if c["folder"] == f]
        fm = [c for c in fc if c["mine"]]
        rep(f"  {f:8s} {len(fm):,}/{len(fc):,} ({len(fm) / max(len(fc), 1):.0%})")
    dens = [c["density"] for c in chunks if c["n_tokens"] >= ccfg["mine_min_tokens"]]
    rep("density deciles (chunks >= min tokens): "
        + " ".join(f"{pct(dens, q / 10):.2f}" for q in range(1, 10)))

    def show(c):
        t = c["text"].replace("\n", " ")
        rep(f"  [{c['density']:.2f} | topic {c['topic_score']}] {c['chunk_id']} ({c['n_tokens']} tok) "
            f"{' > '.join(c['section_path'])[:70]}")
        rep(f"     signals: {c['density_signals']}")
        rep(f"     {t[:260]}{' …' if len(t) > 260 else ''}")

    eligible = [c for c in chunks if c["n_tokens"] >= ccfg["mine_min_tokens"]
                and c["doc_id"] not in ccfg["exclude_from_mining"]]
    ranked = sorted(eligible, key=lambda c: c["density"], reverse=True)
    rep("\n-- Top 10 density --")
    for c in ranked[:10]:
        show(c)
    rep("\n-- Bottom 5 density --")
    for c in ranked[-5:]:
        show(c)
    rng = random.Random(seed)
    near = [c for c in eligible if abs(c["density"] - ccfg["density_threshold"]) < 0.03]
    rep(f"\n-- 3 random chunks near the threshold ({ccfg['density_threshold']} ± 0.03) --")
    for c in rng.sample(near, min(3, len(near))):
        show(c)

    rep("\n-- Readiness-dimension coverage preview (mineable chunks, keyword hits >= "
        f"{cfg['dimensions']['min_hits']}) --")
    dim = Counter(d for c in mine for d in c["dimensions_preview"])
    for name in cfg["dimensions"]["keywords"]:
        n = dim[name]
        rep(f"  {name:24s} {n:6,} ({n / max(len(mine), 1):.0%})  docs={len({c['doc_id'] for c in mine if name in c['dimensions_preview']})}")
    rep(f"  {'(none)':24s} {sum(not c['dimensions_preview'] for c in mine):6,}")
    rep("\nAPI cost this phase: $0 (heuristic density, no LLM calls)")
    rep.save()


if __name__ == "__main__":
    main()
