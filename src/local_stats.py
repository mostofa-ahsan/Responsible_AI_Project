"""S6 statistics for the local evaluation. CPU only, no API calls.

Builds the per-item master table (judge grade, Opus grade where it exists, key-fact support / contradiction,
Phase 2c claim checks, token F1, ROUGE-L, answer length) and computes per (system, split):
  judge strict accuracy (correct / N), lenient accuracy (correct + 0.5 partial), hallucination rate,
  key-fact recall (primary checker), recall (secondary checker), contradiction rate, any-contradiction,
  contradicted-claim rate, unsupported-claim rate, token F1, ROUGE-L, mean answer words
with 95% bootstrap CIs (2,000 resamples). Paired tests vs the same family's base model on identical items:
exact McNemar on binary correct and a paired bootstrap on every metric's difference (p = share of centred
bootstrap means at least as extreme), Holm-Bonferroni within each metric across the 36 comparisons
(12 fine-tuned systems x 3 splits), effect size Cohen's d_z of the paired differences (and Cohen's g for McNemar).

Outputs (results/local_eval/): master_items.csv, metrics_by_system.csv, significance.csv, by_group.csv,
validation.json, keyfact_validation.md, SUMMARY.md
"""

import json
import math

import numpy as np
import pandas as pd

from local_common import (FAMILIES, FAMILY_LABEL, OUT, PER_ITEM, SPLITS, VARIANT_LABEL, VARIANTS, items, log,
                          opus_grades, preds, read_jsonl, sys_id, systems)

N_BOOT = 2000
METRICS = [  # (column, label, higher_is_better)
    ("judge_strict", "Judge strict accuracy", True),
    ("judge_lenient", "Judge accuracy (lenient)", True),
    ("hallucination", "Hallucination rate", False),
    ("keyfact_recall", "Key-fact recall", True),
    ("keyfact_recall_secondary", "Key-fact recall (secondary checker)", True),
    ("contradiction_rate", "Contradiction rate", False),
    ("any_contradiction", "Any contradiction", False),
    ("claim_contradicted_rate", "Contradicted-claim rate (2c)", False),
    ("claim_unsupported_rate", "Unsupported-claim rate (2c)", False),
    ("token_f1", "Token F1", True),
    ("rouge_l", "ROUGE-L", True),
    ("answer_words", "Mean answer words", None),
]
SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def boot_ci(x, seed=0):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return (float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(x))


def paired_boot(d, seed=0):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan, dz=np.nan, n=len(d))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    m = d.mean()
    p = float(((np.abs(means - m) >= abs(m)).sum() + 1) / (N_BOOT + 1))
    sd = d.std(ddof=1)
    return dict(diff=float(m), lo=float(np.percentile(means, 2.5)), hi=float(np.percentile(means, 97.5)), p=p,
                dz=float(m / sd) if sd > 0 else 0.0, n=len(d))


def mcnemar(base_correct, sys_correct):
    from scipy.stats import binomtest
    b = int(np.sum(base_correct & ~sys_correct))
    c = int(np.sum(~base_correct & sys_correct))
    p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c > 0 else 1.0
    g = (c / (b + c) - 0.5) if b + c > 0 else 0.0
    return dict(b_base_only=b, c_sys_only=c, p=p, cohen_g=g)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(np.where(ok, p, np.inf))
    m = ok.sum()
    running = 0.0
    for rank, i in enumerate(idx[:m]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def build_master():
    from eval_closedbook import rouge_l, token_f1
    its = items()
    jg = {(r["split"], r["qa_id"], r["system"]): r for r in read_jsonl(PER_ITEM / "judge_grades.jsonl")}
    choice = json.loads((OUT / "judge_choice.json").read_text()) if (OUT / "judge_choice.json").exists() else {}

    def support(name):
        d = {}
        for r in read_jsonl(PER_ITEM / name):
            d.setdefault((r["split"], r["qa_id"], r["system"]), {})[r["k"]] = r["p"]
        return d
    mc7b, flan = support("support_minicheck7b.jsonl"), support("support_flant5.jsonl")
    nf = {}
    for r in read_jsonl(PER_ITEM / "nli_facts.jsonl"):
        nf.setdefault((r["split"], r["qa_id"], r["system"]), {})[r["k"]] = r["contradicted"]
    nc = {}
    for r in read_jsonl(PER_ITEM / "nli_claims.jsonl"):     # appended incrementally: dedupe by claim index
        nc.setdefault((r["split"], r["qa_id"], r["system"]), {})[r["i"]] = r
    nc = {k: list(v.values()) for k, v in nc.items()}
    claims = {(r["split"], r["qa_id"], r["system"]): len(r["claims"]) for r in read_jsonl(PER_ITEM / "claims.jsonl")}
    kf = {}
    from local_common import KEYFACTS
    for s in SPLITS:
        for r in read_jsonl(KEYFACTS / f"{s}.jsonl"):
            kf[(s, r["qa_id"])] = len(r["facts"])
    n_pairs = sum(kf.values()) * len(systems())
    primary = "minicheck7b" if len([1 for v in mc7b.values() for _ in v]) >= 0.9 * max(n_pairs, 1) else "flant5"
    prim, sec = (mc7b, flan) if primary == "minicheck7b" else (flan, mc7b)
    rows = []
    for fam, var in systems():
        sid = sys_id(fam, var)
        for s in SPLITS:
            p = preds(fam, var, s)
            og_all = opus_grades(fam, var, s)
            for q, it in its[s].items():
                if q not in p:
                    continue
                k = (s, q, sid)
                g = jg.get(k, {})
                nfact = kf.get((s, q))

                def recall(src):
                    v = src.get(k)
                    if not nfact or not v or len(v) < nfact:
                        return np.nan
                    return sum(x >= 0.5 for x in v.values()) / nfact
                con = nf.get(k)
                con_ok = bool(nfact and con and len(con) >= nfact)
                cl = nc.get(k)
                ncl = claims.get(k)
                cl_ok = cl is not None and ncl is not None and len(cl) == ncl and ncl > 0
                lg = g.get("local_grade")
                og = og_all.get(q, {}).get("verdict")
                rows.append({
                    "system": sid, "family": fam, "variant": var, "split": s, "qa_id": q,
                    "q_type": it.get("q_type"), "dimension": it.get("dimension"), "difficulty": it.get("difficulty"),
                    "answer_words": len(p[q].split()), "token_f1": token_f1(p[q], it["answer"]),
                    "rouge_l": rouge_l(p[q], it["answer"]),
                    "local_grade": lg, "judge_strict": float(lg == "correct") if lg else np.nan,
                    "judge_lenient": SCORE[lg] if lg else np.nan,
                    "hallucination": float(bool(g.get("hallucinated_specific"))) if lg else np.nan,
                    "opus_grade": og, "opus_strict": float(og == "correct") if og else np.nan,
                    "opus_lenient": SCORE[og] if og else np.nan,
                    "opus_unsupported_claims": og_all.get(q, {}).get("unsupported_claims"),
                    "n_facts": nfact or np.nan, "keyfact_recall": recall(prim), "keyfact_recall_secondary": recall(sec),
                    "contradiction_rate": (sum(con.values()) / nfact) if con_ok else np.nan,
                    "any_contradiction": float(any(con.values())) if con_ok else np.nan,
                    "n_claims": ncl if ncl is not None else np.nan,
                    "claim_contradicted_rate": (sum(c["contradicted"] for c in cl) / ncl) if cl_ok else np.nan,
                    "claim_unsupported_rate": (sum(not c["supported"] for c in cl) / ncl) if cl_ok else np.nan,
                })
    df = pd.DataFrame(rows)
    meta = {"primary_checker": primary, "secondary_checker": "flant5" if primary == "minicheck7b" else "minicheck7b",
            "judge": choice.get("judge_name"), "judge_prompt": choice.get("prompt"),
            "judge_exploratory": choice.get("exploratory", True)}
    return df, meta


def metrics_table(df):
    out = []
    for (sid, s), g in df.groupby(["system", "split"], sort=False):
        row = {"system": sid, "family": g.family.iloc[0], "variant": g.variant.iloc[0], "split": s, "n_items": len(g)}
        for col, _, _ in METRICS:
            m, lo, hi, n = boot_ci(g[col].tolist())
            row[col] = m
            row[col + "_lo"] = lo
            row[col + "_hi"] = hi
            row[col + "_n"] = n
        if g.opus_grade.notna().any():
            for col in ("opus_strict", "opus_lenient"):
                m, lo, hi, n = boot_ci(g[col].tolist())
                row[col], row[col + "_lo"], row[col + "_hi"] = m, lo, hi
        out.append(row)
    return pd.DataFrame(out)


def significance(df):
    out = []
    for fam in FAMILIES:
        for var in VARIANTS[1:]:
            for s in SPLITS:
                a = df[(df.system == sys_id(fam, "base")) & (df.split == s)].set_index("qa_id")
                b = df[(df.system == sys_id(fam, var)) & (df.split == s)].set_index("qa_id")
                common = a.index.intersection(b.index)
                if len(common) == 0:
                    continue
                a, b = a.loc[common], b.loc[common]
                for col, _, _ in METRICS:
                    d = (b[col] - a[col]).to_numpy(dtype=float)
                    r = paired_boot(d)
                    out.append({"family": fam, "variant": var, "split": s, "metric": col, "test": "paired_bootstrap",
                                "base_mean": float(np.nanmean(a[col])) if a[col].notna().any() else np.nan,
                                "sys_mean": float(np.nanmean(b[col])) if b[col].notna().any() else np.nan,
                                "diff": r["diff"], "diff_lo": r["lo"], "diff_hi": r["hi"], "p": r["p"],
                                "effect": r["dz"], "effect_type": "cohen_dz", "n": r["n"]})
                ok = a.local_grade.notna() & b.local_grade.notna()
                if ok.sum():
                    m = mcnemar((a.judge_strict[ok] == 1).to_numpy(), (b.judge_strict[ok] == 1).to_numpy())
                    out.append({"family": fam, "variant": var, "split": s, "metric": "judge_strict", "test": "mcnemar",
                                "base_mean": float(a.judge_strict[ok].mean()), "sys_mean": float(b.judge_strict[ok].mean()),
                                "diff": float(b.judge_strict[ok].mean() - a.judge_strict[ok].mean()),
                                "diff_lo": np.nan, "diff_hi": np.nan, "p": m["p"], "effect": m["cohen_g"],
                                "effect_type": f"cohen_g (b={m['b_base_only']}, c={m['c_sys_only']})", "n": int(ok.sum())})
    sig = pd.DataFrame(out)
    sig["p_holm"] = np.nan
    for (metric, test), g in sig.groupby(["metric", "test"]):
        sig.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    return sig


def by_group(df):
    out = []
    for gcol in ("q_type", "dimension", "difficulty"):
        for (sid, s, gv), g in df.groupby(["system", "split", gcol]):
            r = {"system": sid, "family": g.family.iloc[0], "variant": g.variant.iloc[0], "split": s,
                 "group_type": gcol, "group": gv, "n": len(g)}
            for col in ("judge_lenient", "judge_strict", "keyfact_recall", "contradiction_rate", "token_f1", "opus_lenient"):
                r[col] = float(g[col].mean()) if g[col].notna().any() else np.nan
            out.append(r)
    return pd.DataFrame(out)


def validation(df, mt):
    from scipy.stats import pearsonr, pointbiserialr, spearmanr
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    v = {}
    op = df[df.opus_grade.notna()]
    v["n_opus_items"] = int(len(op))

    def auc(y, x):
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 10 or len(set(y[m])) < 2:
            return None
        return round(float(roc_auc_score(y[m], x[m])), 4)

    def pb(x, y):
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 10 or len(set(x[m])) < 2 or len(set(y[m])) < 2:
            return None
        r, p = pointbiserialr(x[m], y[m])
        return {"r": round(float(r), 4), "p": float(p), "n": int(m.sum())}
    y_cor = (op.opus_grade == "correct").to_numpy(dtype=float)
    y_inc = (op.opus_grade == "incorrect").to_numpy(dtype=float)
    for col in ("keyfact_recall", "keyfact_recall_secondary"):
        x = op[col].to_numpy(dtype=float)
        v[f"auc_{col}_vs_opus_correct"] = auc(y_cor, x)
        v[f"auc_{col}_vs_opus_not_incorrect"] = auc(1 - y_inc, x)
    for col in ("any_contradiction", "contradiction_rate", "claim_contradicted_rate", "hallucination"):
        v[f"pointbiserial_{col}_vs_opus_incorrect"] = pb(op[col].to_numpy(dtype=float), y_inc)
    seen = op[op.opus_unsupported_claims.notna()]
    if len(seen):
        yu = (seen.opus_unsupported_claims != "0").to_numpy(dtype=float)
        for col in ("any_contradiction", "hallucination", "claim_unsupported_rate"):
            v[f"pointbiserial_{col}_vs_opus_unsupported_any (seen facts)"] = pb(seen[col].to_numpy(dtype=float), yu)
    m = df.local_grade.notna() & df.opus_grade.notna()
    if m.sum():
        v["item_kappa_local_vs_opus_all_items"] = round(float(cohen_kappa_score(df.local_grade[m], df.opus_grade[m])), 4)
    # system level (18 cells with Opus grades)
    cells = mt[mt.opus_lenient.notna()] if "opus_lenient" in mt.columns else None
    for col in ("judge_lenient", "keyfact_recall", "contradiction_rate", "token_f1", "rouge_l"):
        if cells is None:
            break
        c = cells[[col, "opus_lenient"]].dropna()
        if len(c) >= 3:
            v[f"system_{col}_vs_opus_lenient"] = {"pearson": round(float(pearsonr(c[col], c.opus_lenient)[0]), 4),
                                                  "spearman": round(float(spearmanr(c[col], c.opus_lenient)[0]), 4),
                                                  "n_cells": int(len(c))}
    # checker agreement
    pr, se = df.keyfact_recall.to_numpy(dtype=float), df.keyfact_recall_secondary.to_numpy(dtype=float)
    mm = ~(np.isnan(pr) | np.isnan(se))
    if mm.sum() > 10:
        v["checker_answer_level_recall_pearson"] = round(float(pearsonr(pr[mm], se[mm])[0]), 4)
    pa = {(r["split"], r["qa_id"], r["system"], r["k"]): r["p"] for r in read_jsonl(PER_ITEM / "support_minicheck7b.jsonl")}
    pb_ = {(r["split"], r["qa_id"], r["system"], r["k"]): r["p"] for r in read_jsonl(PER_ITEM / "support_flant5.jsonl")}
    ks = [k for k in pa if k in pb_]
    if len(ks) > 10:
        a = np.array([pa[k] for k in ks])
        b = np.array([pb_[k] for k in ks])
        v["checker_fact_level_minicheck7b_vs_flant5"] = {
            "n_pairs": len(ks), "agreement": round(float(((a >= .5) == (b >= .5)).mean()), 4),
            "kappa": round(float(cohen_kappa_score(a >= .5, b >= .5)), 4),
            "pearson_prob": round(float(pearsonr(a, b)[0]), 4),
            "support_rate_minicheck7b": round(float((a >= .5).mean()), 4),
            "support_rate_flant5": round(float((b >= .5).mean()), 4)}
    cov = df.claim_contradicted_rate.notna()
    v["phase2c_coverage"] = {"answers_with_claim_checks": int(cov.sum()), "answers_total": int(len(df))}
    return v


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def cell(r, col, d=3):
    m = r.get(col)
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "–"
    return f"{m:.{d}f} [{r[col + '_lo']:.{d}f}, {r[col + '_hi']:.{d}f}]"


def write_md(mt, sig, v, meta):
    def lookup(fam, var, s):
        r = mt[(mt.family == fam) & (mt.variant == var) & (mt.split == s)]
        return r.iloc[0].to_dict() if len(r) else None
    expl = meta["judge_exploratory"]
    L = ["# Local evaluation: summary", "",
         "_Generated by `src/local_stats.py`; every number comes from `metrics_by_system.csv` (95% bootstrap CIs, "
         "2,000 resamples). No API calls._", "",
         f"Judge: **{meta['judge']}** (prompt {meta['judge_prompt']}), calibrated against Opus 5.5 "
         "(see `judge_calibration.md`). "
         + ("**The judge did not pass the calibration gate: judge accuracy and hallucination rate are EXPLORATORY; "
            "key-fact recall and contradiction rate are the primary metrics.**" if expl else
            "It passed the calibration gate, so judge accuracy is a primary metric alongside key-fact recall."),
         f"Key-fact support checker: **{meta['primary_checker']}** (secondary: {meta['secondary_checker']}); "
         "contradiction: DeBERTa-v3-large NLI.", "",
         "Rows: base model, the 1-epoch model of run_2026-10-03 (cosine over 1 epoch), and epochs 1–3 of the "
         "3-epoch run (cosine over 3 epochs). `*` = significantly different from the same family's base after "
         "Holm correction (paired bootstrap; McNemar for strict accuracy), p_holm < 0.05.", ""]
    sigset = {(r.family, r.variant, r.split, r.metric) for r in sig.itertuples()
              if r.test == "paired_bootstrap" and r.p_holm < 0.05}
    cols = [("judge_lenient", "Judge acc" + (" (expl.)" if expl else "")), ("hallucination", "Halluc."),
            ("keyfact_recall", "KF recall"), ("contradiction_rate", "Contra."), ("token_f1", "F1"), ("rouge_l", "ROUGE-L")]
    for s in SPLITS:
        L += [f"## {s}", "", "| Model | Variant | " + " | ".join(c[1] for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        for fam in FAMILIES:
            for var in VARIANTS:
                r = lookup(fam, var, s)
                if not r:
                    continue
                cs = []
                for c, _ in cols:
                    val = fmt(r.get(c))
                    if (fam, var, s, c) in sigset:
                        val += "*"
                    cs.append(val)
                L.append(f"| {FAMILY_LABEL[fam]} | {VARIANT_LABEL[var]} | " + " | ".join(cs) + " |")
        L.append("")
    L += ["## Method validation (vs Opus, 6 systems with Opus grades)", "",
          "| check | value |", "|---|---|"]
    for k, val in v.items():
        L.append(f"| {k} | {json.dumps(val) if isinstance(val, dict) else val} |")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")

    K = ["# Key-fact recall and contradiction (method 2): validation against Opus", "",
         "_Generated by `src/local_stats.py` from `validation.json`._", "",
         f"Key facts: 1–5 atomic facts per reference answer, decomposed by the local judge model from the reference only "
         f"(evidence as context). Support: {meta['primary_checker']} (premise = model answer, claim = fact, supported if "
         "p ≥ 0.5); secondary checker in the second recall column. Contradiction: DeBERTa-v3-large NLI (premise = answer, "
         "sentence windows if > 450 tokens, max over windows; contradicted if p_contra ≥ 0.5 and > p_entail).", "",
         "| validation | value |", "|---|---|"]
    for k, val in v.items():
        K.append(f"| {k} | {json.dumps(val) if isinstance(val, dict) else val} |")
    K += ["", "Interpretation guide: AUC is the probability that a random Opus-correct answer has higher key-fact "
          "recall than a random non-correct one (0.5 = chance). Point-biserial r > 0 for contradiction vs Opus "
          "incorrect means contradictions concentrate in answers Opus marks wrong. Phase 2c: 'unsupported' claims are "
          "not necessarily false (the model may state true outside knowledge); only 'contradicted' counts as an error."]
    (OUT / "keyfact_validation.md").write_text("\n".join(K) + "\n")


def main():
    lg = log()
    df, meta = build_master()
    lg.info(f"[S6] master table: {len(df)} answers; primary checker {meta['primary_checker']}; "
            f"judge graded {int(df.local_grade.notna().sum())}")
    df.to_csv(OUT / "master_items.csv", index=False)
    mt = metrics_table(df)
    sig = significance(df)
    # attach vs-base tests to the metrics table
    piv = sig[sig.test == "paired_bootstrap"].pivot_table(index=["family", "variant", "split"], columns="metric",
                                                         values=["diff", "p", "p_holm", "effect"], aggfunc="first")
    piv.columns = [f"{m}_vs_base_{stat}" for stat, m in piv.columns]
    mc = sig[sig.test == "mcnemar"].set_index(["family", "variant", "split"])[["p", "p_holm", "effect"]]
    mc.columns = ["judge_strict_mcnemar_p", "judge_strict_mcnemar_p_holm", "judge_strict_mcnemar_cohen_g"]
    mt = mt.merge(piv.reset_index(), on=["family", "variant", "split"], how="left") \
           .merge(mc.reset_index(), on=["family", "variant", "split"], how="left")
    mt.to_csv(OUT / "metrics_by_system.csv", index=False)
    sig.to_csv(OUT / "significance.csv", index=False)
    by_group(df).to_csv(OUT / "by_group.csv", index=False)
    v = validation(df, mt)
    v["meta"] = meta
    (OUT / "validation.json").write_text(json.dumps(v, indent=2, default=float))
    write_md(mt, sig, {k: x for k, x in v.items() if k != "meta"}, meta)
    lg.info("[S6] wrote metrics_by_system.csv, significance.csv, by_group.csv, validation.json, SUMMARY.md, "
            "keyfact_validation.md")


if __name__ == "__main__":
    main()
