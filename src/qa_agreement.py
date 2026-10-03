"""Agreement between human review verdicts and the automatic filters.

Reads a review CSV (default data/qa_pairs/pilot_v2_review.csv), maps human_verdict to
accept/reject (accept: accept, yes, y, 1, pass, ok, good, keep; reject: reject, no, n, 0, fail,
bad, drop), and compares it with passed_filters: confusion matrix, observed agreement, Cohen's
kappa, and every disagreement with the filter's reason and the reviewer's note. Rows without a
verdict are ignored.

Usage:
    python src/qa_agreement.py [--run pilot_v2]
"""

import argparse
import csv

from qa_common import run_paths
from utils import Report, load_config

ACCEPT = {"accept", "accepted", "yes", "y", "1", "pass", "ok", "good", "keep", "true"}
REJECT = {"reject", "rejected", "no", "n", "0", "fail", "bad", "drop", "false"}


def cohen_kappa(tp, fn, fp, tn):
    n = tp + fn + fp + tn
    if n == 0:
        return float("nan")
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="pilot_v2")
    args = ap.parse_args()
    cfg = load_config()
    path = run_paths(cfg, args.run)["review"]
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labeled, unknown = [], []
    for r in rows:
        v = r.get("human_verdict", "").strip().lower()
        if not v:
            continue
        if v in ACCEPT or v in REJECT:
            labeled.append((r, v in ACCEPT, r["passed_filters"].strip().lower() == "true"))
        else:
            unknown.append((r["qa_id"], v))

    rep = Report(f"qa_agreement_{args.run}", cfg)
    rep(f"=== Human vs automatic filters: {path.name} ===")
    rep(f"rows: {len(rows)}; with a verdict: {len(labeled)}; unrecognized verdicts: {unknown[:10]}")
    if not labeled:
        rep("no labeled rows yet")
        rep.save()
        return
    tp = sum(h and a for _, h, a in labeled)      # both accept
    fn = sum(h and not a for _, h, a in labeled)  # human accepts, filter rejects
    fp = sum(not h and a for _, h, a in labeled)  # human rejects, filter accepts
    tn = sum(not h and not a for _, h, a in labeled)
    n = len(labeled)
    rep("\nconfusion matrix (rows = human, cols = filter)")
    rep(f"{'':16s} {'filter accept':>14s} {'filter reject':>14s}")
    rep(f"{'human accept':16s} {tp:>14d} {fn:>14d}")
    rep(f"{'human reject':16s} {fp:>14d} {tn:>14d}")
    rep(f"\nobserved agreement {(tp + tn) / n:.1%}; Cohen's kappa {cohen_kappa(tp, fn, fp, tn):.3f}")
    rep(f"filter precision (accepted pairs the human also accepts): {tp / max(tp + fp, 1):.1%}; "
        f"filter recall of human-accepted pairs: {tp / max(tp + fn, 1):.1%}")
    if "repaired" in rows[0]:
        rep_rows = [(r, h, a) for r, h, a in labeled if r["repaired"].strip().lower() == "true"]
        if rep_rows:
            ok = sum(h for _, h, a in rep_rows if a)
            rep(f"repaired pairs accepted by the filter: human accepts {ok}/{sum(a for _, _, a in rep_rows)}")
    rep("\n-- disagreements --")
    for r, h, a in labeled:
        if h != a:
            rep(f"\n[{r['qa_id']}] human={'accept' if h else 'reject'} filter={'accept' if a else 'reject'}"
                f" {r.get('reject_reason', '')}")
            rep(f"Q: {r['question']}")
            rep(f"A: {r['answer']}")
            if r.get("human_notes", "").strip():
                rep(f"note: {r['human_notes']}")
            if r.get("judge_rationale", "").strip():
                rep(f"judge: {r['judge_rationale'][:300]}")
    rep.save()


if __name__ == "__main__":
    main()
