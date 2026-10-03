"""LLM check of the heuristic density score in the borderline band (density_scorer model).

Scores every eligible chunk (>= mine_min_tokens, topic gate passed, not excluded) whose
heuristic density is within --band of chunk.density_threshold, on a 1-5 scale for its value
for source-grounded QA about responsible AI in higher education. Reports how many below-
threshold chunks it would promote (score >= --promote) and how many above-threshold chunks it
would demote (score <= --demote). Report only: mine flags in chunks.jsonl are not changed.

Writes data/chunks/density_llm.jsonl and logs/density_llm_report.txt. Calls are cached.

Usage:
    python src/density_llm.py [--band 0.1] [--limit N] [--workers 8]
"""

import argparse
import json
import random
from collections import Counter

from pydantic import BaseModel, Field

from chunk import read_jsonl
from llm import LLM
from qa_common import run_parallel
from utils import Report, get_logger, load_config, repo_path

SYSTEM = """You rate passages from books and journal articles for building a question-answering \
dataset about responsible AI in higher education (AI ethics, governance and policy, teaching and \
assessment with AI, AI literacy, faculty readiness, privacy, equity, institutional strategy).

Score 1-5 how much substantive, citable knowledge the passage contains for that dataset:
5 = dense with definitions, findings, frameworks, recommendations or explained mechanisms on the topic
4 = several solid, on-topic facts or arguments worth asking about
3 = some useful content mixed with filler, or useful but only loosely on-topic
2 = mostly narrative, signposting, examples without takeaways, or off-topic technical detail
1 = no usable knowledge (front/back matter, lists of names, boilerplate, unrelated content)"""

PROMPT = """Source: {title}
Section: {section}

<passage>
{text}
</passage>"""


class Rating(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = Field(description="One short sentence")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--band", type=float, default=0.1)
    ap.add_argument("--promote", type=int, default=4)
    ap.add_argument("--demote", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    cc = cfg["chunk"]
    log = get_logger("density_llm", cfg)
    chunks = read_jsonl(repo_path(cfg["paths"]["chunks"]) / "chunks.jsonl")
    th = cc["density_threshold"]
    band = [c for c in chunks if c["n_tokens"] >= cc["mine_min_tokens"]
            and c["doc_id"] not in cc["exclude_from_mining"]
            and c["topic_score"] >= cc.get("min_topic_score", 0)
            and abs(c["density"] - th) <= args.band]
    if args.limit:
        band = random.Random(0).sample(band, min(args.limit, len(band)))
    log.info(f"{len(band)} chunks in band {th - args.band:.2f}-{th + args.band:.2f}")
    llm = LLM(cfg, "density_llm")

    def work(c):
        return llm.complete(PROMPT.format(title=c["title"], section=" > ".join(c["section_path"]),
                                          text=c["text"]), SYSTEM, role="density_scorer",
                            json_schema=Rating)

    rows, failures = [], []
    for c, res in run_parallel(work, band, args.workers, log):
        if isinstance(res, Exception):
            failures.append(c["chunk_id"])
            continue
        rows.append({"chunk_id": c["chunk_id"], "density": c["density"], "mine": c["mine"],
                     "llm_score": res.parsed.score, "reason": res.parsed.reason, "model": res.served_model})
    out = repo_path(cfg["paths"]["chunks"]) / "density_llm.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["chunk_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    below = [r for r in rows if r["density"] < th]
    above = [r for r in rows if r["density"] >= th]
    promote = [r for r in below if r["llm_score"] >= args.promote]
    demote = [r for r in above if r["llm_score"] <= args.demote]
    by_id = {c["chunk_id"]: c for c in chunks}
    rep = Report("density_llm", cfg)
    rep(f"=== Haiku density check: {len(rows)} borderline chunks "
        f"(heuristic density {th - args.band:.2f}-{th + args.band:.2f}, threshold {th}) ===")
    rep(f"model: {cfg['models']['density_scorer']['model']}; failures: {len(failures)}")
    rep(f"below threshold ({len(below)}): LLM score distribution {dict(sorted(Counter(r['llm_score'] for r in below).items()))}")
    rep(f"above threshold ({len(above)}): LLM score distribution {dict(sorted(Counter(r['llm_score'] for r in above).items()))}")
    rep(f"\nwould PROMOTE into mining (below threshold, score >= {args.promote}): {len(promote)} chunks, "
        f"{sum(by_id[r['chunk_id']]['n_tokens'] for r in promote):,} tokens")
    rep(f"would DEMOTE (above threshold, score <= {args.demote}): {len(demote)} chunks, "
        f"{sum(by_id[r['chunk_id']]['n_tokens'] for r in demote):,} tokens")
    cur = sum(c["mine"] for c in chunks)
    rep(f"mineable chunks: {cur:,} now -> {cur + len(promote):,} with promotions "
        f"-> {cur + len(promote) - len(demote):,} with promotions and demotions")
    rng = random.Random(1)
    for title, group in (("promote", promote), ("demote", demote)):
        rep(f"\n-- 4 random {title} examples --")
        for r in rng.sample(group, min(4, len(group))):
            c = by_id[r["chunk_id"]]
            rep(f"  [{r['density']:.2f} -> LLM {r['llm_score']}] {r['chunk_id']}: {r['reason']}")
            rep(f"     {c['text'][:220].replace(chr(10), ' ')} …")
    usage = [json.loads(line) for line in repo_path(cfg["llm"]["usage_log"]).open(encoding="utf-8")]
    us = [u for u in usage if u["stage"] == "density_llm" and not u["cached"]]
    rep(f"\ncost: {len(us)} calls, in={sum(u['input'] for u in us):,} out={sum(u['output'] for u in us):,} "
        f"${sum(u['cost'] for u in us):.4f}")
    rep.save()


if __name__ == "__main__":
    main()
