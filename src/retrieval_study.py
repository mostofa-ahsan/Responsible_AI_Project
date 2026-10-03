"""Retrieval study: which setup finds the source chunk for real pilot questions?

Queries are the passed questions of a QA run (default pilot_v2). A retrieved chunk counts as
relevant if it is the pair's source chunk or contains the pair's longest evidence span (the
~100-token overlap means a span can sit in two adjacent chunks).

Variants (all over the full 6,406-chunk corpus):
  a  dense, chunk embedded with "Title/Section" header, query with the Qwen3 instruction
  b  dense, chunk embedded without the header (embeddings cached in data/index/)
  c  BM25 (bm25s, English stemmer + stopwords) over header + text
  d  hybrid: reciprocal-rank fusion (k=60) of BM25 and the better dense variant
  e  hybrid top-50 reranked with Qwen3-Reranker-4B
Reports recall@1/5/10 and MRR@10, overall and by q_type, plus latency per query.

Usage:
    python src/retrieval_study.py [--run pilot_v2] [--rerank-depth 50]
"""

import argparse
import gc
import json
import time
from collections import defaultdict

import numpy as np

from embed import embed_text, encode, load_model
from qa_common import load_chunks, read_jsonl, run_paths, ws
from utils import Report, get_logger, load_config, repo_path

RERANK_INSTRUCTION = "Given a question about responsible AI in higher education, retrieve passages that answer it"


def metrics(ranked, relevant, ks=(1, 5, 10)):
    out = {f"R@{k}": float(np.mean([bool(set(r[:k]) & rel) for r, rel in zip(ranked, relevant)])) for k in ks}
    rr = []
    for r, rel in zip(ranked, relevant):
        pos = next((i for i, c in enumerate(r[:10]) if c in rel), None)
        rr.append(0.0 if pos is None else 1.0 / (pos + 1))
    out["MRR@10"] = float(np.mean(rr))
    return out


def rrf(lists, k=60, depth=100):
    score = defaultdict(float)
    for lst in lists:
        for rank, cid in enumerate(lst[:depth]):
            score[cid] += 1.0 / (k + rank + 1)
    return [c for c, _ in sorted(score.items(), key=lambda x: -x[1])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="pilot_v2")
    ap.add_argument("--rerank-depth", type=int, default=50)
    args = ap.parse_args()

    import bm25s
    import faiss
    import Stemmer
    import torch

    cfg = load_config()
    ecfg = cfg["embed"]
    log = get_logger("retrieval_study", cfg)
    idx_dir = repo_path(cfg["paths"]["index"])
    chunks = load_chunks(cfg)
    ids = [r["chunk_id"] for r in json.loads((idx_dir / "ids.json").read_text())]
    pairs = [p for p in read_jsonl(run_paths(cfg, args.run)["final"]) if p["passed_filters"]]
    queries = [p["question"] for p in pairs]
    relevant = []
    for p in pairs:
        span = ws(max(p["evidence"].split(" … "), key=len))
        rel = {p["chunk_id"]} | {cid for cid in ids if span and span in ws(chunks[cid]["text"])}
        relevant.append(rel)
    log.info(f"{len(queries)} queries; mean relevant chunks per query {np.mean([len(r) for r in relevant]):.2f}")

    results, timing, ranked = {}, {}, {}
    K = 100

    # --- dense a / b ------------------------------------------------------
    model = load_model(ecfg)
    t0 = time.time()
    qv = encode(model, queries, ecfg["batch_size"], prompt=ecfg["query_instruction"])
    q_secs = (time.time() - t0) / len(queries)
    emb_a = np.load(idx_dir / "embeddings.npy")
    nh_path = idx_dir / "embeddings_noheader.npy"
    nh_ids_path = idx_dir / "ids_noheader.json"
    if nh_path.exists() and json.loads(nh_ids_path.read_text()) == ids:
        emb_b = np.load(nh_path)
    else:
        log.info("embedding all chunks without header (one-off, cached)")
        emb_b = encode(model, [embed_text(chunks[c], False) for c in ids], ecfg["batch_size"])
        np.save(nh_path, emb_b)
        nh_ids_path.write_text(json.dumps(ids))
    for name, emb in (("a dense+header", emb_a), ("b dense, no header", emb_b)):
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        t0 = time.time()
        _, hits = index.search(qv, K)
        timing[name] = q_secs + (time.time() - t0) / len(queries)
        ranked[name] = [[ids[i] for i in row] for row in hits]
        results[name] = metrics(ranked[name], relevant)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- BM25 c -------------------------------------------------------------
    stemmer = Stemmer.Stemmer("english")
    corpus = [embed_text(chunks[c], True) for c in ids]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer, show_progress=False),
                    show_progress=False)
    t0 = time.time()
    docs, _ = retriever.retrieve(bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False),
                                 k=K, show_progress=False)
    timing["c BM25"] = (time.time() - t0) / len(queries)
    ranked["c BM25"] = [[ids[i] for i in row] for row in docs]
    results["c BM25"] = metrics(ranked["c BM25"], relevant)

    # --- hybrid d -------------------------------------------------------------
    best_dense = max(("a dense+header", "b dense, no header"), key=lambda n: (results[n]["R@5"], results[n]["MRR@10"]))
    name_d = f"d hybrid RRF (BM25 + {best_dense[0]})"
    ranked[name_d] = [rrf([b, d]) for b, d in zip(ranked["c BM25"], ranked[best_dense])]
    timing[name_d] = timing["c BM25"] + timing[best_dense]
    results[name_d] = metrics(ranked[name_d], relevant)

    # --- rerank e -------------------------------------------------------------
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("Qwen/Qwen3-Reranker-4B", device="cuda",
                            model_kwargs={"torch_dtype": torch.bfloat16},
                            prompts={"rag": RERANK_INSTRUCTION}, default_prompt_name="rag",
                            max_length=2048)
    name_e = f"e hybrid + Qwen3-Reranker-4B (top {args.rerank_depth})"
    t0 = time.time()
    out = []
    for q, cand in zip(queries, ranked[name_d]):
        cand = cand[:args.rerank_depth]
        scores = reranker.predict([(q, embed_text(chunks[c], True)) for c in cand], batch_size=8,
                                  show_progress_bar=False)
        out.append([c for _, c in sorted(zip(scores, cand), key=lambda x: -x[0])])
    timing[name_e] = timing[name_d] + (time.time() - t0) / len(queries)
    ranked[name_e] = out
    results[name_e] = metrics(out, relevant)

    rep = Report(f"retrieval_study_{args.run}", cfg)
    rep(f"=== Retrieval study: {len(queries)} passed {args.run} questions over {len(ids):,} chunks ===")
    rep("relevant = source chunk or any chunk containing the pair's longest evidence span "
        f"(mean {np.mean([len(r) for r in relevant]):.2f} relevant chunks per query)")
    rep(f"\n{'variant':52s} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR@10':>7} {'s/query':>8}")
    for name, m in results.items():
        rep(f"{name:52s} {m['R@1']:6.1%} {m['R@5']:6.1%} {m['R@10']:6.1%} {m['MRR@10']:7.3f} {timing[name]:8.3f}")
    rep("\n-- R@5 by q_type --")
    types = sorted({p["q_type"] for p in pairs})
    rep(f"{'variant':52s} " + " ".join(f"{t[:11]:>12}" for t in types))
    for name in results:
        cells = []
        for t in types:
            sel = [i for i, p in enumerate(pairs) if p["q_type"] == t]
            r5 = np.mean([bool(set(ranked[name][i][:5]) & relevant[i]) for i in sel])
            cells.append(f"{r5:7.0%} (n={len(sel):2d})")
        rep(f"{name:52s} " + " ".join(f"{c:>12}" for c in cells))
    (repo_path(cfg["paths"]["logs"]) / f"retrieval_study_{args.run}.json").write_text(
        json.dumps({"results": results, "timing": timing}, indent=2))
    rep.save()


if __name__ == "__main__":
    main()
