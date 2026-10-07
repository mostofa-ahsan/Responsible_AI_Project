"""Statistics and STATUS for the matching-base regeneration (python src/regen.py stats|status). Local, CPU only.

One per-item table of every system (results/regen/master_items.csv.gz):
  <fam>__<var>              adapter served on its training base (NF4-dequantized bf16): the MAIN numbers
  <fam>__<var>__bf16serve   the same adapter served on the bf16 base (earlier results, untouched; ablation)
  <fam>__base, __concise40  base model (bf16, no adapter); __nf4base: base on the dequantized NF4 weights (control)
  <fam>__base__rerender, __concise40__rerender  Gemma prompts re-rendered with the training chat function
Every number written here is computed from result files.
"""

import json
import math
import shutil
import time

import numpy as np
import pandas as pd

from local_common import FAMILY_LABEL, KEYFACTS, OUT as LOCAL_OUT, PER_ITEM as LOCAL_PI, log, read_jsonl
from regen import ANS, ORDER, PI, RG, TESTS, adapter_path, all_items, cpt_epochs
from utils import repo_path

TE = repo_path("results/trained_eval")
CPT = repo_path("results/cpt")
SR = repo_path("results/seed_replication")
SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
METRICS = ["judge_lenient", "judge_strict", "hallucination", "keyfact_recall", "keyfact_recall_flan",
           "contradiction_rate", "token_f1", "rouge_l", "exact_repro", "answer_words"]
TEST_LABEL = {"test_trained_exact": "Exact training questions", "test_seen_facts": "Reworded training questions",
              "test_indomain": "In-domain (unseen chunks)", "test_heldout_docs": "Held-out documents"}


def label(var, serving):
    nice = {"base": "base", "concise40": "concise base (≤ 40 words)", "nf4base": "base on NF4 weights (control)",
            "ft1run": "QA-only 1-epoch run", "seed43_ep1": "QA-only ep1, seed 43"}
    if var.startswith("ep"):
        s = f"QA-only {var}"
    elif var.startswith("cpt_ep"):
        s = f"mixed CPT ep{var[6:]}"
    else:
        s = nice.get(var, var)
    return s + {"bf16serve": " [bf16-served]", "rerender": " [re-rendered]", "match": ""}.get(serving, "")


def keyed(paths, field, by="k"):
    d = {}
    for p in paths:
        for r in read_jsonl(p):
            d.setdefault((r["split"], r["qa_id"], r["system"]), {})[r[by]] = r[field]
    return d


def load_tables():
    sup = keyed([LOCAL_PI / "support_minicheck7b.jsonl", TE / "per_item" / "support_minicheck7b.jsonl",
                 CPT / "per_item" / "support_minicheck7b.jsonl", PI / "support_minicheck7b.jsonl"] +
                [SR / f / "seed43_epoch1" / "support_minicheck7b.jsonl" for f in ORDER], "p")
    flan = keyed([LOCAL_PI / "support_flant5.jsonl", CPT / "per_item" / "support_flant5.jsonl",
                  PI / "support_flant5.jsonl"] + [SR / f / "seed43_epoch1" / "support_flant5.jsonl" for f in ORDER], "p")
    con = keyed([LOCAL_PI / "nli_facts.jsonl", TE / "per_item" / "nli_facts.jsonl", CPT / "per_item" / "nli_facts.jsonl",
                 PI / "nli_facts.jsonl"] + [SR / f / "seed43_epoch1" / "nli_facts.jsonl" for f in ORDER], "contradicted")
    judge = {}
    for p in (LOCAL_PI / "judge_grades.jsonl", TE / "per_item" / "judge_grades.jsonl", CPT / "per_item" / "judge_grades.jsonl",
              PI / "judge_grades.jsonl"):
        for r in read_jsonl(p):
            if r.get("local_grade"):
                judge[(r["split"], r["qa_id"], r["system"])] = r
    return sup, flan, con, judge


def systems():
    """[(family, variant, serving, answers {split: {qa: text}}, source system id used in the per-item files)]"""
    import cpt
    out = []
    for fam in ORDER:
        for var in ("base", "concise40"):
            out.append((fam, var, "match", {s: cpt.preds(fam, var, s) for s in TESTS}, f"{fam}__{var}"))
        olds = ["ep1", "ep2", "ep3"] + [f"cpt_ep{e}" for e in cpt_epochs(fam)] + ["ft1run"]
        for var in olds:
            out.append((fam, var, "bf16serve", {s: cpt.preds(fam, var, s) if not (var == "ft1run" and s == "test_trained_exact")
                                                else {r["qa_id"]: r["prediction"] for r in read_jsonl(TE / "answers" / f"{fam}__ft1run" / f"{s}_predictions.jsonl")}
                                                for s in TESTS}, f"{fam}__{var}"))
        sd = SR / fam / "seed43_epoch1"
        out.append((fam, "seed43_ep1", "bf16serve", {s: {r["qa_id"]: r["prediction"] for r in read_jsonl(sd / f"{s}_predictions.jsonl")}
                                                     for s in TESTS}, f"{fam}__seed43_ep1"))
        for d in sorted(ANS.glob(f"{fam}__*")):
            _, var, suf = d.name.split("__")
            serving = "match" if suf == "mb" else "rerender"
            out.append((fam, var if var != "nf4base" else "nf4base", serving,
                        {s: {r["qa_id"]: r["prediction"] for r in read_jsonl(d / f"{s}_predictions.jsonl")} for s in TESTS},
                        d.name))
    return out


def per_item():
    from eval_closedbook import rouge_l, token_f1
    its = all_items()
    kf = {(s, r["qa_id"]): len(r["facts"]) for s in TESTS for r in read_jsonl(KEYFACTS / f"{s}.jsonl")}
    sup, flan, con, judge = load_tables()
    rows = []
    for fam, var, serving, ans, src in systems():
        name = f"{fam}__{var}" + ("" if serving == "match" else f"__{serving}")
        for s in TESTS:
            for q, a in ans[s].items():
                it = its[s].get(q)
                if it is None:
                    continue
                k = (s, q, src)
                n = kf.get((s, q))
                sp, fl, cn = sup.get(k, {}), flan.get(k, {}), con.get(k, {})
                g = judge.get(k, {}).get("local_grade")
                rl = rouge_l(a, it["answer"])
                rows.append({"system": name, "family": fam, "variant": var, "serving": serving, "split": s, "qa_id": q,
                             "q_type": it.get("q_type"), "dimension": it.get("dimension"), "difficulty": it.get("difficulty"),
                             "doc_id": it.get("doc_id"), "paraphrase_test_id": it.get("paraphrase_test_id"),
                             "answer_words": len(a.split()), "token_f1": token_f1(a, it["answer"]), "rouge_l": rl,
                             "exact_repro": float(rl >= 0.8), "local_grade": g,
                             "judge_lenient": SC[g] if g else np.nan, "judge_strict": float(g == "correct") if g else np.nan,
                             "hallucination": float(bool(judge.get(k, {}).get("hallucinated_specific"))) if g else np.nan,
                             "keyfact_recall": sum(v >= 0.5 for v in sp.values()) / n if n and len(sp) >= n else np.nan,
                             "keyfact_recall_flan": sum(v >= 0.5 for v in fl.values()) / n if n and len(fl) >= n else np.nan,
                             "contradiction_rate": sum(cn.values()) / n if n and len(cn) >= n else np.nan,
                             "contradicted_facts": sum(cn.values()) if n and len(cn) >= n else np.nan,
                             "supported_facts": sum(v >= 0.5 for v in sp.values()) if n and len(sp) >= n else np.nan,
                             "n_facts": n, "answer": a, "reference": it["answer"], "question": it["question"]})
    return pd.DataFrame(rows)


def best_variant(mt, fam, prefix):
    """Best epoch (matching base) = highest mean judge accuracy over the 4 test types."""
    c = mt[(mt.family == fam) & (mt.serving == "match") & mt.variant.str.match(prefix + r"\d+$")]
    if c.empty or c.judge_lenient.isna().all():
        return None
    return c.groupby("variant").judge_lenient.mean().idxmax()


def cmd_stats(a):
    import local_pack as P
    from local_stats import boot_ci, holm, mcnemar, paired_boot
    df = per_item()
    df.to_csv(RG / "master_items.csv.gz", index=False, compression="gzip")
    rows = []
    for (sys_, s), g in df.groupby(["system", "split"], sort=False):
        r = {"system": sys_, "family": g.family.iloc[0], "variant": g.variant.iloc[0], "serving": g.serving.iloc[0],
             "split": s, "n": len(g)}
        for c in METRICS:
            m, lo, hi, n = boot_ci(g[c].tolist())
            r.update({c: m, c + "_lo": lo, c + "_hi": hi, c + "_n": n})
        rows.append(r)
    mt = pd.DataFrame(rows)
    mt.to_csv(RG / "metrics_by_system.csv", index=False)
    best = {f: {"qa": best_variant(mt, f, "ep"), "cpt": best_variant(mt, f, "cpt_ep")} for f in ORDER}
    (RG / "best_epochs.json").write_text(json.dumps(best, indent=2))

    def comp(a_sys, b_sys, kind, fam, var):
        out = []
        for s in TESTS:
            A = df[(df.system == a_sys) & (df.split == s)].set_index("qa_id")
            B = df[(df.system == b_sys) & (df.split == s)].set_index("qa_id")
            c = A.index.intersection(B.index)
            if not len(c):
                continue
            A, B = A.loc[c], B.loc[c]
            for m in METRICS:
                pb = paired_boot((B[m] - A[m]).to_numpy(dtype=float))
                out.append({"comparison": kind, "family": fam, "variant": var, "reference": a_sys, "system": b_sys,
                            "split": s, "metric": m, "test": "paired_bootstrap", "ref_mean": float(np.nanmean(A[m])) if A[m].notna().any() else np.nan,
                            "sys_mean": float(np.nanmean(B[m])) if B[m].notna().any() else np.nan, "diff": pb["diff"],
                            "diff_lo": pb["lo"], "diff_hi": pb["hi"], "p": pb["p"], "effect_dz": pb["dz"], "n": pb["n"]})
            ok = A.judge_strict.notna() & B.judge_strict.notna()
            if ok.sum():
                mc = mcnemar((A.judge_strict[ok] == 1).to_numpy(), (B.judge_strict[ok] == 1).to_numpy())
                out.append({"comparison": kind, "family": fam, "variant": var, "reference": a_sys, "system": b_sys, "split": s,
                            "metric": "judge_strict", "test": "mcnemar", "diff": float(B.judge_strict[ok].mean() - A.judge_strict[ok].mean()),
                            "p": mc["p"], "effect_dz": mc["cohen_g"], "n": int(ok.sum())})
        return out
    sig = []
    present = set(df.system)
    for fam in ORDER:
        for var in sorted(set(df[(df.family == fam)].variant)):
            m, b = f"{fam}__{var}", f"{fam}__{var}__bf16serve"
            if var not in ("base", "concise40") and m in present:
                sig += comp(f"{fam}__base", m, "vs_base", fam, var)
            if m in present and b in present:
                sig += comp(b, m, "matching_vs_bf16serve", fam, var)
            if var.startswith("cpt_ep") and m in present and f"{fam}__ep{var[6:]}" in present:
                sig += comp(f"{fam}__ep{var[6:]}", m, "cpt_vs_qa_only", fam, var)
        if f"{fam}__nf4base" in present:
            sig += comp(f"{fam}__base", f"{fam}__nf4base", "nf4base_vs_base", fam, "nf4base")
        if f"{fam}__concise40" in present:
            sig += comp(f"{fam}__base", f"{fam}__concise40", "vs_base", fam, "concise40")
            for k in ("qa", "cpt"):
                if best[fam][k] and f"{fam}__{best[fam][k]}" in present:
                    sig += comp(f"{fam}__concise40", f"{fam}__{best[fam][k]}", f"best_{k}_vs_concise", fam, best[fam][k])
    sig = pd.DataFrame(sig)
    if len(sig):
        sig["p_holm"] = np.nan
        for _, g in sig.groupby(["comparison", "metric", "test"]):
            sig.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    sig.to_csv(RG / "significance.csv", index=False)
    sp = sig[(sig.comparison == "matching_vs_bf16serve") & (sig.test == "paired_bootstrap") &
             sig.metric.isin(["judge_lenient", "keyfact_recall", "exact_repro"])] if len(sig) else sig
    sp.to_csv(RG / "serving_precision.csv", index=False)
    gap = []
    ex = df[(df.split == "test_trained_exact") & df.paraphrase_test_id.notna()]
    pa = df[df.split == "test_seen_facts"]
    for sys_, g in ex.groupby("system"):
        p2 = pa[pa.system == sys_].set_index("qa_id")
        g = g[g.paraphrase_test_id.isin(p2.index)]
        if g.empty:
            continue
        for m in ("judge_lenient", "keyfact_recall", "exact_repro"):
            d = g[m].to_numpy(dtype=float) - p2.loc[g.paraphrase_test_id, m].to_numpy(dtype=float)
            pb = paired_boot(d)
            gap.append({"system": sys_, "family": g.family.iloc[0], "variant": g.variant.iloc[0], "serving": g.serving.iloc[0],
                        "metric": m, "exact": float(np.nanmean(g[m])) if g[m].notna().any() else np.nan,
                        "reworded": float(np.nanmean(p2.loc[g.paraphrase_test_id, m])) if g[m].notna().any() else np.nan,
                        "gap": pb["diff"], "lo": pb["lo"], "hi": pb["hi"], "p": pb["p"], "n": pb["n"]})
    pd.DataFrame(gap).to_csv(RG / "robustness_gap.csv", index=False)
    lc = []
    for fam in ORDER:
        for s in TESTS:
            for v in ["base", "concise40", best[fam]["qa"], best[fam]["cpt"]]:
                if not v:
                    continue
                r = mt[(mt.system == f"{fam}__{v}") & (mt.split == s)]
                if len(r):
                    r = r.iloc[0]
                    lc.append({"family": fam, "split": s, "variant": v, "label": label(v, "match"), "answer_words": r.answer_words,
                               "judge_lenient": r.judge_lenient, "keyfact_recall": r.keyfact_recall,
                               "contradiction_rate": r.contradiction_rate, "token_f1": r.token_f1})
    pd.DataFrame(lc).to_csv(RG / "length_control.csv", index=False)
    write_tables(P, mt, sig, best)
    write_section(mt, sig, best)
    P.build()
    log().info(f"[R stats] {len(df)} answers, {len(mt)} system x test rows, {len(sig)} tests")


def f3(x, d=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_tables(P, mt, sig, best):
    dag = set()
    if len(sig):
        for r in sig[(sig.test == "paired_bootstrap") & (sig.p_holm < 0.05)].itertuples():
            dag.add((r.system, r.split, r.metric, r.comparison))
    cols = [("judge_lenient", "Judge acc", True), ("keyfact_recall", "KF recall", True), ("contradiction_rate", "Contra.", False),
            ("token_f1", "F1", True), ("exact_repro", "Exact repro.", True)]
    for s in TESTS:
        rows, vals = [], []
        for fam in ORDER:
            vs = ["base", "concise40", "ep1", "ep2", "ep3"] + [f"cpt_ep{e}" for e in cpt_epochs(fam)] + ["nf4base"]
            for v in vs:
                name = f"{fam}__{v}"
                r = mt[(mt.system == name) & (mt.split == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                cells, vv = [], []
                for c, _, _ in cols:
                    mk = ("†" if (name, s, c, "vs_base") in dag else "") + ("‡" if (name, s, c, "cpt_vs_qa_only") in dag else "") + \
                         ("§" if (name, s, c, "matching_vs_bf16serve") in dag else "")
                    cells.append(f3(r[c]) + mk)
                    vv.append(r[c])
                rows.append([FAMILY_LABEL[fam], label(v, "match")] + cells)
                vals.append(vv)
        bold = set()
        if vals:
            arr = np.array(vals, dtype=float)
            for j, (_, _, hib) in enumerate(cols):
                if np.all(np.isnan(arr[:, j])):
                    continue
                best_ = np.nanmax(arr[:, j]) if hib else np.nanmin(arr[:, j])
                bold |= {(int(i), j + 2) for i in np.where(np.isclose(arr[:, j], best_))[0]}
        P.write_table(f"T7_matching_{s.replace('test_', '')}", f"Table 7. {TEST_LABEL[s]}: adapters served on their training base",
                      ["Model", "System"] + [c[1] for c in cols], rows, bold,
                      note="Adapters are served on the NF4-dequantized base they were trained on (QLoRA). † differs from base, "
                           "‡ mixed CPT differs from QA-only at the same epoch, § differs from the same adapter served on the bf16 "
                           "base (paired bootstrap, Holm-adjusted p < 0.05). '–' = not measured.")
    sp = sig[(sig.comparison == "matching_vs_bf16serve") & (sig.test == "paired_bootstrap")] if len(sig) else sig
    rows = []
    for r in sp[sp.metric.isin(["judge_lenient", "keyfact_recall", "exact_repro"])].sort_values(["family", "variant", "split", "metric"]).itertuples():
        rows.append([FAMILY_LABEL[r.family], label(r.variant, "match"), r.split.replace("test_", ""), r.metric,
                     f3(r.ref_mean), f3(r.sys_mean), f"{r.diff:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]",
                     f"{r.p_holm:.4f}" if r.p_holm == r.p_holm else "–"])
    P.write_table("T11_serving_precision", "Table 11. Serving precision: adapter on the bf16 base vs on its training (NF4) base",
                  ["Model", "Adapter", "Test", "Metric", "bf16 base", "Training base", "Δ [95% CI]", "p (Holm)"], rows,
                  note="Same adapters, same pre-tokenized greedy decoding except the base weights: bf16 (earlier results) vs the "
                       "NF4-dequantized bf16 weights the adapter was trained on. Paired bootstrap on identical items.")


def write_section(mt, sig, best):
    L = ["## Serving precision: adapters on their training base", "",
         "QLoRA adapters are trained on an NF4-quantized base. Served on the original bf16 base (all earlier results), they lose "
         "part of what they learned (results/diagnostics/format_check.md). All adapters were re-served on the NF4-dequantized "
         "base (verified per model: answer loss within 2% of NF4) with prompts pre-tokenized by the training chat function. "
         "These are the main numbers below; the bf16-served results are kept as an ablation (Table 11)."]
    for fam in ORDER:
        v = json.loads((RG / f"dequant_verify_{fam}.json").read_text()) if (RG / f"dequant_verify_{fam}.json").exists() else None
        if v:
            L.append(f"- {FAMILY_LABEL[fam]}: answer loss NF4 {v['answer_loss_nf4']:.4f} vs dequantized {v['answer_loss_dequantized_bf16']:.4f} "
                     f"({v['relative_difference']:.2%}); best QA-only epoch {best[fam]['qa']}, best mixed-CPT epoch {best[fam]['cpt']}.")
    for s in TESTS:
        parts = []
        for fam in ORDER:
            seg = []
            for v in ("base", best[fam]["qa"], best[fam]["cpt"]):
                r = mt[(mt.system == f"{fam}__{v}") & (mt.split == s)] if v else mt.iloc[0:0]
                if len(r):
                    seg.append(f"{label(v, 'match')} {f3(r.judge_lenient.iloc[0])}")
            if seg:
                parts.append(f"{FAMILY_LABEL[fam]}: " + ", ".join(seg))
        L.append(f"- {TEST_LABEL[s]} (judge accuracy): " + "; ".join(parts) + ".")
    (RG / "RESULTS_section.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ STATUS

def cmd_status(a):
    st = RG / "stage_status.tsv"
    rows = [l.split("\t") for l in st.read_text().splitlines()] if st.exists() else []
    dl = dict(l.split("=") for l in (RG / "deadlines.env").read_text().split()) if (RG / "deadlines.env").exists() else {}
    fmt = lambda e: time.strftime("%F %T", time.localtime(int(e))) if e else "?"
    L = [f"# Matching-base regeneration + research_materials: STATUS", "", f"_Written {time.strftime('%Y-%m-%d %H:%M')}. Start "
         f"{fmt(dl.get('START'))}; checkpoint {fmt(dl.get('CHECKPOINT'))}; hard limit {fmt(dl.get('HARD'))}._", ""]
    over = [r for r in rows if r and r[0] == "power_limit" and float(r[2]) > 270]
    if over:
        L = ["> **WARNING: GPU power limit above 270 W at launch** (cap resets on reboot; cannot be set from WSL).", ""] + L
    L += ["| Stage | Result | Minutes | Note |", "|---|---|---|---|"]
    last = {}
    for r in rows:
        if r and r[0].startswith(("R", "deployment")) and len(r) > 1 and r[1] in ("done", "failed", "pending"):
            last[r[0]] = r
    for k, r in last.items():
        L.append(f"| {k} | {r[1]} | {r[2] if r[1] != 'pending' else '–'} | {r[3] if len(r) > 3 else ''} |")
    L += ["", "## Dequantized-base verification (QA-only ep3 adapter, 20 training items, tolerance 2%)", ""]
    for fam in ORDER:
        p = RG / f"dequant_verify_{fam}.json"
        if p.exists():
            v = json.loads(p.read_text())
            L.append(f"- {fam}: NF4 {v['answer_loss_nf4']} vs dequantized {v['answer_loss_dequantized_bf16']} "
                     f"(rel {v['relative_difference']:.3%}) -> {'PASS' if v['passed'] else 'FAIL'}")
        else:
            L.append(f"- {fam}: not built yet")
    import regen
    L += ["", "## Coverage per priority tier", ""]
    graded = {(r["split"], r["qa_id"], r["system"]) for r in read_jsonl(PI / "judge_grades.jsonl")}
    checked = {(r["split"], r["qa_id"], r["system"]) for r in read_jsonl(PI / "support_minicheck7b.jsonl")}
    for t, desc in (("P1", "Qwen + Gemma QA-only ep1-3"), ("P2", "Llama QA-only ep1-3"), ("P3", "mixed-CPT adapters"),
                    ("P4", "base on NF4 weights (control)"), ("D", "1-epoch run, seed 43, Gemma base/concise re-render")):
        sysl = regen.tier_systems(t)
        ans = sum(len(regen.preds(f, v, s_, sp)) for f, v, s_ in sysl for sp in TESTS)
        g = sum(1 for f, v, s_ in sysl for sp in TESTS for q in regen.preds(f, v, s_, sp) if (sp, q, regen.sid(f, v, s_)) in graded)
        c = sum(1 for f, v, s_ in sysl for sp in TESTS for q in regen.preds(f, v, s_, sp) if (sp, q, regen.sid(f, v, s_)) in checked)
        L.append(f"- {t} ({desc}): {len(sysl)} systems, {ans:,} answers, {g:,} judge-graded, {c:,} with MiniCheck-7B checks")
    L += ["", "## Decisions, pending work and resume commands", "",
          "- Disk: judge (15 GB) and MiniCheck-7B (15.5 GB) cannot coexist with a 10 GB reserve, so to have P1-P2 graded AND "
          "checked by the checkpoint the swap runs twice: judge -> MiniCheck for P1-P2, then judge -> MiniCheck again for "
          "the rest (two extra ~2 min downloads).",
          "- Earlier results are untouched and relabelled '__bf16serve' in the reports (adapter on the bf16 base)."]
    for r in rows:
        if len(r) > 3 and r[1] == "pending":
            L.append(f"- PENDING {r[0]}: {r[3]}")
    L += ["- Resume anything unfinished: `bash scripts/resume_regen_after_crash.sh` (skips finished stages). Single steps: "
          "`python src/regen.py dequant|gen --model <m>`, `python src/regen.py grade|checks --tier P12|rest`, "
          "`python src/regen.py stats`, `python src/build_research_materials.py`.",
          "- Deployment variants on the matching base (merged AWQ/RTN/GGUF): not run; they need the dequantized checkpoint, "
          "llm-compressor (src/trained_quant.py with --model models_dequant/<m>) and a further judge + MiniCheck swap."]
    g = repo_path("logs/gpu_temp.log")
    if g.exists() and dl.get("START"):
        start = time.strftime("%F %T", time.localtime(int(dl["START"])))
        t = [(float(p[1]), float(p[2]), p[0]) for p in (l.split(",") for l in g.read_text(errors="ignore").splitlines()[1:])
             if len(p) == 6 and p[0] >= start and p[1].replace(".", "").isdigit()]
        if t:
            L.append(f"- GPU since start: max {max(x[0] for x in t):.0f} °C, max power draw {max(x[1] for x in t):.0f} W.")
    L.append(f"- Disk free now: {shutil.disk_usage(str(RG)).free / 1e9:.1f} GB.")
    pushes = [r for r in rows if r and r[0] == "push"]
    L.append(f"- Failed push attempts: {len(pushes)}.")
    (RG / "STATUS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
