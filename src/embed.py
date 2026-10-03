"""Phase 3b: embed all chunks and build a FAISS index in data/index/.

Embeds every chunk in data/chunks/chunks.jsonl (not only mine=true; Phase 4's
retrieval filter and the RAG baselines search the whole corpus) with the
configured embedding model, L2-normalizes, and writes:

    data/index/embeddings.npy   float32 [N, dim], row i <-> ids[i]
    data/index/ids.json         chunk_ids in row order, with a hash of the embedded text
    data/index/chunks.faiss     IndexFlatIP over the normalized vectors (cosine)
    data/index/meta.json        model, dim, query instruction, counts

Resumable: rows whose chunk_id and text hash are unchanged are reused; only new
or changed chunks are embedded. Queries must be encoded with
embed.query_instruction (see search()).

Usage:
    python src/embed.py [--limit N] [--force] [--check 200]
"""

import argparse
import hashlib
import json
import random
import re
import time

import numpy as np

from chunk import read_jsonl
from utils import Report, get_logger, load_config, repo_path


def embed_text(c, include_header):
    if not include_header:
        return c["text"]
    section = " > ".join(c["section_path"])
    return f"Title: {c['title']}\nSection: {section}\n\n{c['text']}"


def text_hash(t):
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


def load_model(ecfg):
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        ecfg["model"], device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": getattr(torch, ecfg["dtype"])},
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = ecfg["max_seq_length"]
    return model


def encode(model, texts, batch_size, prompt=None):
    # Sort by length so batches pad little; restore order afterwards.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    vecs = model.encode([texts[i] for i in order], batch_size=batch_size, prompt=prompt,
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    out = np.empty_like(vecs)
    out[order] = vecs
    return out.astype(np.float32)


def search(index, model, ecfg, queries, k=5):
    """Encode queries with the instruction prefix and return (scores, row ids)."""
    q = encode(model, queries, ecfg["batch_size"], prompt=ecfg["query_instruction"])
    return index.search(q, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, help="only embed the first N chunks needing it")
    ap.add_argument("--force", action="store_true", help="re-embed everything")
    ap.add_argument("--check", type=int, default=200,
                    help="self-retrieval sanity check on N random chunks (0 to skip)")
    args = ap.parse_args()

    import faiss

    cfg = load_config()
    ecfg = cfg["embed"]
    log = get_logger("embed", cfg)
    idx_dir = repo_path(cfg["paths"]["index"])
    idx_dir.mkdir(parents=True, exist_ok=True)
    chunks = read_jsonl(repo_path(cfg["paths"]["chunks"]) / "chunks.jsonl")
    texts = {c["chunk_id"]: embed_text(c, ecfg["include_header"]) for c in chunks}
    hashes = {cid: text_hash(t) for cid, t in texts.items()}

    old = {}
    meta_path, ids_path, emb_path = idx_dir / "meta.json", idx_dir / "ids.json", idx_dir / "embeddings.npy"
    if not args.force and emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("model") == ecfg["model"] and meta.get("include_header") == ecfg["include_header"]:
            ids = json.loads(ids_path.read_text())
            emb = np.load(emb_path)
            old = {r["chunk_id"]: (r["hash"], emb[i]) for i, r in enumerate(ids)}

    todo = [cid for cid in texts if cid not in old or old[cid][0] != hashes[cid]]
    if args.limit:
        todo = todo[:args.limit]
    log.info(f"{len(chunks)} chunks; {len(chunks) - len(todo)} cached, {len(todo)} to embed")

    model, secs = None, 0.0
    new_vecs = {}
    if todo:
        model = load_model(ecfg)
        t0 = time.time()
        vecs = encode(model, [texts[c] for c in todo], ecfg["batch_size"])
        secs = time.time() - t0
        new_vecs = dict(zip(todo, vecs))
        log.info(f"embedded {len(todo)} chunks in {secs:.0f}s")

    # Assemble in chunks.jsonl order; chunks still missing (when --limit) are left out.
    rows, mats = [], []
    for cid in texts:
        if cid in new_vecs:
            v = new_vecs[cid]
        elif cid in old and old[cid][0] == hashes[cid]:
            v = old[cid][1]
        else:
            continue
        rows.append({"chunk_id": cid, "hash": hashes[cid]})
        mats.append(v)
    emb = np.vstack(mats).astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    np.save(emb_path.with_suffix(".tmp.npy"), emb)
    emb_path.with_suffix(".tmp.npy").replace(emb_path)
    ids_path.write_text(json.dumps(rows))
    faiss.write_index(index, str(idx_dir / "chunks.faiss"))
    meta = {"model": ecfg["model"], "dim": int(emb.shape[1]), "n": int(emb.shape[0]),
            "metric": "inner product on L2-normalized vectors (cosine)",
            "include_header": ecfg["include_header"], "query_instruction": ecfg["query_instruction"],
            "max_seq_length": ecfg["max_seq_length"], "dtype": ecfg["dtype"]}
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"index: {index.ntotal} vectors, dim {emb.shape[1]}")

    rep = Report("embed", cfg)
    rep(f"=== Phase 3b embeddings: {index.ntotal:,}/{len(chunks):,} chunks indexed ===")
    rep(f"model: {ecfg['model']} ({ecfg['dtype']}, max_seq_length {ecfg['max_seq_length']}), dim {emb.shape[1]}")
    rep(f"this run: embedded {len(todo):,} chunks in {secs:.0f}s; reused {index.ntotal - len(todo):,}")
    rep(f"files: {emb_path.name} ({emb_path.stat().st_size / 1e6:.0f} MB), chunks.faiss "
        f"({(idx_dir / 'chunks.faiss').stat().st_size / 1e6:.0f} MB), IndexFlatIP (exact cosine)")
    norms = np.linalg.norm(emb, axis=1)
    rep(f"vector norms: min {norms.min():.4f} max {norms.max():.4f}")

    if args.check and index.ntotal:
        # Self-retrieval: a middle sentence of a chunk, as a query, should find its chunk.
        model = model or load_model(ecfg)
        rng = random.Random(0)
        by_id = {c["chunk_id"]: c for c in chunks}
        row_of = {r["chunk_id"]: i for i, r in enumerate(rows)}
        sample = rng.sample([r["chunk_id"] for r in rows if by_id[r["chunk_id"]]["chunk_type"] == "text"],
                            min(args.check, index.ntotal))
        queries = []
        for cid in sample:
            sents = [s for s in re.split(r"(?<=[.!?])\s+", by_id[cid]["text"]) if 60 <= len(s) <= 300]
            queries.append(sents[len(sents) // 2] if sents else by_id[cid]["text"][:200])
        _, hits = search(index, model, ecfg, queries, k=5)
        top1 = sum(hits[i][0] == row_of[cid] for i, cid in enumerate(sample))
        top5 = sum(row_of[cid] in hits[i] for i, cid in enumerate(sample))
        rep(f"self-retrieval check ({len(sample)} chunks, middle sentence as query): "
            f"top-1 {top1 / len(sample):.0%}, top-5 {top5 / len(sample):.0%}")

        probe = "What should a university AI governance policy include?"
        scores, hits = search(index, model, ecfg, [probe], k=5)
        rep(f"\nprobe query: {probe!r}")
        for s, i in zip(scores[0], hits[0]):
            c = by_id[rows[i]["chunk_id"]]
            rep(f"  {s:.3f}  {c['chunk_id']}  {' > '.join(c['section_path'])[:70]}")
    rep.save()


if __name__ == "__main__":
    main()
