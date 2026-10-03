"""Compare a candidate judge with the Opus judge on an existing run's judged pairs.

Re-judges, with the candidate role (default judge_fast = Haiku), exactly the pair versions the
Opus judge saw in <run>: each first-pass pair that reached the judge, and each repaired version.
Pass/fail uses the run's thresholds (grounding >= min_grounding, standalone, value >= min_value).
Reports pass/fail agreement, Cohen's kappa, per-criterion agreement and cost. Calls are cached.

Usage:
    python src/judge_compare.py [--run pilot_v3] [--role judge_fast]
"""

import argparse
import json
from collections import Counter

from llm import LLM
from qa_agreement import cohen_kappa
from qa_common import read_jsonl, run_parallel, run_paths
from qa_filter import PROMPT, SYSTEM, Judgement
from utils import Report, get_logger, load_config, repo_path


def verdict(js, q):
    return (js["grounding"] >= q["min_grounding"] and js["standalone"]
            and js.get("value", 5) >= q.get("min_value", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="pilot_v3")
    ap.add_argument("--role", default="judge_fast")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    q = cfg["qa"]
    log = get_logger("judge_compare", cfg)
    paths = run_paths(cfg, args.run)
    gen = {g["qa_id"]: g for g in read_jsonl(paths["generated"])}
    final = read_jsonl(paths["final"])

    items = []   # (label, pair_version, opus_scores)
    for r in final:
        fjs = r["first_pass"]["judge_scores"]
        if "grounding" in fjs:
            items.append((f"{r['qa_id']}#first", gen[r["qa_id"]], fjs))
        if r["repair"] and "grounding" in r["judge_scores"]:
            items.append((f"{r['qa_id']}#repaired", r, r["judge_scores"]))
    llm = LLM(cfg, f"judge_compare:{args.run}")

    def work(it):
        _, p, _ = it
        paras = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(p["paraphrases"])) or "(none)"
        return llm.complete(PROMPT.format(question=p["question"], paraphrases=paras, answer=p["answer"],
                                          evidence=p["evidence"]), SYSTEM, role=args.role, json_schema=Judgement)

    res = {}
    for it, out in run_parallel(work, items, args.workers, log):
        res[it[0]] = out
    rows = []
    for label, _, ojs in items:
        out = res[label]
        if isinstance(out, Exception):
            continue
        c = out.parsed
        cjs = {"grounding": c.grounding_score, "standalone": c.standalone, "value": c.value_score}
        rows.append((label, verdict(ojs, q), verdict(cjs, q), ojs, cjs))

    n = len(rows)
    tp = sum(o and c for _, o, c, _, _ in rows)
    fn = sum(o and not c for _, o, c, _, _ in rows)
    fp = sum(not o and c for _, o, c, _, _ in rows)
    tn = sum(not o and not c for _, o, c, _, _ in rows)
    model = cfg["models"][args.role]["model"]
    rep = Report(f"judge_compare_{args.run}", cfg)
    rep(f"=== {model} vs Opus judge on {n} judged pair versions from {args.run} ===")
    rep(f"pass/fail agreement: {(tp + tn) / max(n, 1):.1%}   Cohen's kappa: {cohen_kappa(tp, fn, fp, tn):.3f}")
    rep(f"{'':14s} {'cand pass':>10s} {'cand fail':>10s}")
    rep(f"{'opus pass':14s} {tp:>10d} {fn:>10d}")
    rep(f"{'opus fail':14s} {fp:>10d} {tn:>10d}")
    rep(f"candidate passes what Opus fails (false accepts): {fp};  candidate fails what Opus passes: {fn}")
    for crit, f in (("grounding>=4", lambda js: js["grounding"] >= q["min_grounding"]),
                    ("standalone", lambda js: js["standalone"]),
                    ("value>=3", lambda js: js.get("value", 5) >= q["min_value"])):
        agree = sum(f(o) == f(c) for _, _, _, o, c in rows)
        rep(f"  {crit:14s} agreement {agree / max(n, 1):.1%}")
    rep(f"grounding pairs (opus, candidate): {Counter((o['grounding'], c['grounding']) for _, _, _, o, c in rows).most_common(8)}")
    # escalation rule from the plan: send to Opus if candidate is uncertain
    unc = [r for r in rows if r[4]["grounding"] < 5 or r[4]["value"] < 4 or not r[4]["standalone"]
           or r[0].endswith("#repaired")]
    missed = [r for r in rows if r not in unc and r[1] != r[2]]
    rep(f"\nescalation rule (candidate grounding<5, value<4, standalone=false, or repaired): "
        f"{len(unc)}/{n} ({len(unc) / max(n, 1):.0%}) would go to Opus; disagreements NOT escalated: {len(missed)}")
    for label, o, c, ojs, cjs in missed[:10]:
        rep(f"  {label}: opus={'pass' if o else 'fail'} {ojs.get('grounding')}/{ojs.get('standalone')}/{ojs.get('value')}  "
            f"cand={'pass' if c else 'fail'} {cjs['grounding']}/{cjs['standalone']}/{cjs['value']}")
    usage = [json.loads(line) for line in repo_path(cfg["llm"]["usage_log"]).open(encoding="utf-8")]
    us = [u for u in usage if u["stage"] == f"judge_compare:{args.run}" and not u["cached"]]
    opus = [u for u in usage if u["stage"] == f"qa_filter:{args.run}" and u["role"] == "judge" and not u["cached"]]
    rep(f"\ncost: candidate ${sum(u['cost'] for u in us):.4f} for {len(us)} calls; "
        f"Opus judge in {args.run} ${sum(u['cost'] for u in opus):.4f} for {len(opus)} calls")
    rep.save()


if __name__ == "__main__":
    main()
