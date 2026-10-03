"""Final report for a full QA run (local files only, no API calls).

Covers: pairs generated / passed, first-pass rate, repaired share, reject reasons,
q_type / difficulty / dimension distributions (generated -> passed), per-document counts,
paraphrase coverage, and spend by stage and role versus the run's budget cap.
Writes logs/<run>_final_report.txt.

Usage:
    python src/run_report.py --run full_v1
"""

import argparse
import json
from collections import Counter, defaultdict

from qa_common import load_metadata, read_jsonl, run_paths
from utils import Report, load_config, repo_path


def pct(a, b):
    return f"{a:,}/{b:,} ({a / b:.1%})" if b else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="full_v1")
    args = ap.parse_args()
    cfg = load_config()
    paths = run_paths(cfg, args.run)
    rows = read_jsonl(paths["final"])
    sel = json.loads(paths["chunks"].read_text())
    status = json.loads(paths["chunks"].with_name(f"{args.run}_chunk_status.json").read_text())
    units = read_jsonl(paths["units"])
    meta = load_metadata(cfg)
    n = len(rows)
    passed = [r for r in rows if r["passed_filters"]]
    reps = [r for r in rows if r["repair"]]
    rep = Report(f"{args.run}_final", cfg)

    rep(f"=== {args.run}: final report ===")
    rep(f"chunks selected {len(sel['chunk_ids']):,}; with units {len({u['chunk_id'] for u in units}):,}; "
        f"used for generation {sum(s['generate'] for s in status.values()):,}")
    rep(f"knowledge units {len(units):,}; valid {pct(sum(u['evidence_valid'] for u in units), len(units))} "
        f"(not verbatim {sum(not u.get('evidence_verbatim', True) for u in units):,}, "
        f"mid-sentence {sum(u.get('evidence_complete') is False for u in units):,})")
    rep(f"pairs generated {n:,}")
    rep(f"PASSED (final) {pct(len(passed), n)}")
    rep(f"passed on first pass {pct(sum(r['first_pass']['passed'] for r in rows), n)}")
    rep(f"repair attempted {pct(len(reps), n)}; passed after repair "
        f"{pct(sum(r['repair']['passed_after_repair'] for r in reps), len(reps))}; "
        f"repaired share of passed pairs {pct(sum(bool(r['repair']) for r in passed), len(passed))}")
    para = Counter(len(r["paraphrases"]) for r in passed)
    rep(f"passed pairs by paraphrase count: 2 = {para[2]:,}, 1 = {para[1]:,}, 0 = {para[0]:,} "
        f"(total training questions incl. paraphrases: {sum(1 + len(r['paraphrases']) for r in passed):,})")

    rep("\n-- Final reject reasons (pairs with each reason) --")
    rc = Counter()
    for r in rows:
        if not r["passed_filters"]:
            rc.update({x.split("(")[0] for x in r["reject_reason"].split(";") if x})
    for k, v in rc.most_common():
        rep(f"  {k:24s} {v:6,} ({v / n:.1%} of generated)")

    for key in ("q_type", "difficulty", "dimension"):
        g, p = Counter(r[key] for r in rows), Counter(r[key] for r in passed)
        rep(f"\n-- {key}: generated -> passed (share of passed) --")
        for k, v in g.most_common():
            rep(f"  {k:24s} {v:6,} -> {p[k]:6,} ({p[k] / max(len(passed), 1):.1%})")

    rep("\n-- Per document (generated -> passed) --")
    by_doc = defaultdict(lambda: [0, 0])
    for r in rows:
        by_doc[r["doc_id"]][0] += 1
        by_doc[r["doc_id"]][1] += r["passed_filters"]
    folder = {}
    for r in read_jsonl(repo_path(cfg["paths"]["chunks"]) / "chunks.jsonl"):
        folder.setdefault(r["doc_id"], r["folder"])
    for f in ("book", "article"):
        docs = [d for d in by_doc if folder.get(d) == f]
        rep(f"{f}s: {len(docs)} docs, {sum(by_doc[d][1] for d in docs):,} passed pairs")
    for d, (g, p) in sorted(by_doc.items(), key=lambda kv: -kv[1][1]):
        rep(f"  {d[:52]:52s} {folder.get(d, '?')[:4]:4s} {g:5,} -> {p:5,} ({p / g:.0%})  "
            f"{meta.get(d, {}).get('title', '')[:50]}")

    usage = [json.loads(line) for line in repo_path(cfg["llm"]["usage_log"]).open(encoding="utf-8")]
    us = [u for u in usage if u["stage"].endswith(f":{args.run}") and not u["cached"]]
    rep("\n-- Spend by stage and role --")
    total = 0.0
    by = defaultdict(lambda: [0, 0, 0, 0.0, 0])
    for u in us:
        b = by[(u["stage"].split(":")[0], u["role"], "batch" if u.get("batch") else "standard")]
        b[0] += 1
        b[1] += u["input"]
        b[2] += u["output"]
        b[3] += u["cost"]
    for (stage, role, mode), (c, i, o, cost, _) in sorted(by.items()):
        total += cost
        rep(f"  {stage:12s} {role:10s} {mode:8s} calls={c:6,} in={i:>11,} out={o:>10,} ${cost:8.2f}")
    cap = (cfg.get("budget", {}).get("run_max_usd") or {}).get(args.run)
    rep(f"  TOTAL ${total:.2f}" + (f" (cap ${cap:.0f}, {total / cap:.0%} used)" if cap else "")
        + f"; ${total / max(len(passed), 1):.4f} per passed pair")
    skipped = sum("repair_skipped_budget" in r["reject_reason"] for r in rows)
    if skipped:
        rep(f"  budget guard: {skipped:,} repairs skipped")
    rep.save()


if __name__ == "__main__":
    main()
