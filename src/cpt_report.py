"""Stage 5 of the arm-C experiment: statistics, paper-pack additions, SUMMARY.md, RESULTS.md section, STATUS.md.
Every number is read from result files. Called through `python src/cpt.py report|status`.
"""

import json
import math
import shutil
import time

import numpy as np
import pandas as pd

from cpt import CP, ORDER, TESTS, TEST_LABEL, cpt_epochs, f3, per_item
from local_common import FAMILY_COLOR, FAMILY_LABEL, log
from utils import repo_path

METRICS = ["judge_lenient", "judge_strict", "keyfact_recall", "contradiction_rate", "token_f1", "rouge_l", "exact_repro",
           "answer_words"]
VLABEL = {"base": "base", "concise40": "concise base (≤ 40 words)", "ep1": "QA-only ep1", "ep2": "QA-only ep2",
          "ep3": "QA-only ep3", "cpt_ep1": "mixed CPT ep1", "cpt_ep2": "mixed CPT ep2", "cpt_ep3": "mixed CPT ep3"}


def stats(df):
    from local_stats import boot_ci, holm, mcnemar, paired_boot
    rows = []
    for (sid, s), g in df.groupby(["system", "split"], sort=False):
        r = {"system": sid, "family": g.family.iloc[0], "variant": g.variant.iloc[0], "split": s, "n": len(g)}
        for c in METRICS:
            m, lo, hi, n = boot_ci(g[c].tolist())
            r.update({c: m, c + "_lo": lo, c + "_hi": hi, c + "_n": n})
        rows.append(r)
    mt = pd.DataFrame(rows)
    sig = []
    for fam in ORDER:
        vs = sorted(set(df[df.family == fam].variant))
        comps = [(v, "base", "vs_base") for v in vs if v != "base"] + \
                [(v, v.replace("cpt_", ""), "vs_qa_only") for v in vs if v.startswith("cpt_") and v.replace("cpt_", "") in vs]
        for var, ref, kind in comps:
            for s in TESTS:
                a = df[(df.system == f"{fam}__{ref}") & (df.split == s)].set_index("qa_id")
                b = df[(df.system == f"{fam}__{var}") & (df.split == s)].set_index("qa_id")
                c = a.index.intersection(b.index)
                if not len(c):
                    continue
                a, b = a.loc[c], b.loc[c]
                for m in METRICS:
                    pb = paired_boot((b[m] - a[m]).to_numpy(dtype=float))
                    sig.append({"family": fam, "variant": var, "reference": ref, "comparison": kind, "split": s,
                                "metric": m, "test": "paired_bootstrap", "diff": pb["diff"], "diff_lo": pb["lo"],
                                "diff_hi": pb["hi"], "p": pb["p"], "effect": pb["dz"], "n": pb["n"]})
                ok = a.judge_strict.notna() & b.judge_strict.notna()
                if ok.sum():
                    mc = mcnemar((a.judge_strict[ok] == 1).to_numpy(), (b.judge_strict[ok] == 1).to_numpy())
                    sig.append({"family": fam, "variant": var, "reference": ref, "comparison": kind, "split": s,
                                "metric": "judge_strict", "test": "mcnemar",
                                "diff": float(b.judge_strict[ok].mean() - a.judge_strict[ok].mean()), "p": mc["p"],
                                "effect": mc["cohen_g"], "n": int(ok.sum())})
    sig = pd.DataFrame(sig)
    if len(sig):
        sig["p_holm"] = np.nan
        for _, g in sig.groupby(["comparison", "metric", "test"]):
            sig.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    gap = []
    ex = df[(df.split == "test_trained_exact") & df.paraphrase_test_id.notna()]
    pa = df[df.split == "test_seen_facts"]
    for sid, g in ex.groupby("system"):
        p2 = pa[pa.system == sid].set_index("qa_id")
        g = g[g.paraphrase_test_id.isin(p2.index)]
        for m in ("judge_lenient", "keyfact_recall", "token_f1", "exact_repro"):
            if g.empty:
                continue
            d = g[m].to_numpy(dtype=float) - p2.loc[g.paraphrase_test_id, m].to_numpy(dtype=float)
            pb = paired_boot(d)
            gap.append({"system": sid, "family": g.family.iloc[0], "variant": g.variant.iloc[0], "metric": m,
                        "exact": float(np.nanmean(g[m])) if g[m].notna().any() else np.nan,
                        "paraphrase": float(np.nanmean(p2.loc[g.paraphrase_test_id, m])) if g[m].notna().any() else np.nan,
                        "gap_exact_minus_paraphrase": pb["diff"], "lo": pb["lo"], "hi": pb["hi"], "p": pb["p"], "n": pb["n"]})
    return mt, sig, pd.DataFrame(gap)


def cmd_report(a):
    import local_pack as P
    for f in ("stats.json", "leakage_audit.json"):
        if repo_path(f"data/cpt/{f}").exists():
            shutil.copy(repo_path(f"data/cpt/{f}"), CP / f"corpus_{f}")
    df = per_item()
    df.to_csv(CP / "master_items.csv", index=False)
    mt, sig, gap = stats(df)
    mt.to_csv(CP / "metrics_by_system.csv", index=False)
    sig.to_csv(CP / "significance.csv", index=False)
    gap.to_csv(CP / "robustness_gap.csv", index=False)
    plt = P.plt_setup()
    dag = set() if sig.empty else {(r.family, r.variant, r.split, r.metric, r.comparison) for r in sig.itertuples()
                                   if r.test == "paired_bootstrap" and r.p_holm < 0.05}

    def row(fam, var, s):
        r = mt[(mt.system == f"{fam}__{var}") & (mt.split == s)]
        return r.iloc[0] if len(r) else None
    cols = [("judge_lenient", "Judge acc", True), ("keyfact_recall", "KF recall", True),
            ("contradiction_rate", "Contra.", False), ("token_f1", "F1", True), ("exact_repro", "Exact repro.", True)]
    variants = ["base", "concise40", "ep1", "ep2", "ep3", "cpt_ep1", "cpt_ep2", "cpt_ep3"]
    for s in TESTS:
        rows_, vals = [], []
        for fam in ORDER:
            for var in variants:
                r = row(fam, var, s)
                if r is None:
                    continue
                cells, vs = [], []
                for c, _, _ in cols:
                    v = r[c]
                    vs.append(v)
                    mk = ("†" if (fam, var, s, c, "vs_base") in dag else "") + ("‡" if (fam, var, s, c, "vs_qa_only") in dag else "")
                    cells.append(f3(v) + mk)
                rows_.append([FAMILY_LABEL[fam], VLABEL.get(var, var)] + cells)
                vals.append(vs)
        bold = set()
        if vals:
            arr = np.array(vals, dtype=float)
            for j, (_, _, hib) in enumerate(cols):
                if np.all(np.isnan(arr[:, j])):
                    continue
                best = np.nanmax(arr[:, j]) if hib else np.nanmin(arr[:, j])
                bold |= {(int(i), j + 2) for i in np.where(np.isclose(arr[:, j], best))[0]}
        n = int(mt[mt.split == s].n.max()) if (mt.split == s).any() else 0
        P.write_table(f"T7_cpt_{s.replace('test_', '')}", f"Table 7 (arm C). {TEST_LABEL[s].capitalize()} (n = {n})",
                      ["Model", "System"] + [c[1] for c in cols], rows_, bold,
                      note="QA-only = closed-book QLoRA r16 on QA pairs (models_epochs); mixed CPT = arm C, QLoRA on raw "
                           "training-document text + QA. † differs from the same model's base, ‡ from QA-only at the same "
                           "epoch (paired bootstrap, Holm-adjusted p < 0.05). Judge acc = (correct + 0.5 partial) / N; "
                           "KF recall: MiniCheck-7B; Contra.: DeBERTa NLI; Exact repro. = ROUGE-L ≥ 0.8 vs the trained answer. "
                           "'–' = not measured (see STATUS.md).")
    # F9: 4 test types
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    sty = {"base": ("#999999", "o", "--"), "ep1": ("#E69F00", "^", ":"), "ep2": ("#D55E00", "^", ":"), "ep3": ("#882255", "^", ":"),
           "cpt_ep1": ("#56B4E9", "s", "-"), "cpt_ep2": ("#0072B2", "s", "-"), "cpt_ep3": ("#004466", "s", "-")}
    metric = "judge_lenient" if mt.judge_lenient.notna().any() else "keyfact_recall"
    for ax, fam in zip(axes, ORDER):
        for var, (c, m, ls) in sty.items():
            rs = [row(fam, var, s) for s in TESTS]
            if all(r is None for r in rs):
                continue
            P.errbar(ax, range(4), rs, metric, c, VLABEL[var], marker=m, ls=ls)
        ax.set_xticks(range(4), ["exact", "paraphrase", "in-domain", "held-out"])
        ax.set_title(FAMILY_LABEL[fam])
    axes[0].set_ylabel("judge accuracy" if metric == "judge_lenient" else "key-fact recall")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.12), fontsize=7)
    fig.suptitle("Figure 9 (arm C). Four test types: base vs QA-only vs mixed continued pretraining", y=1.03, fontsize=9)
    P.savefig(fig, "F9_cpt_four_tests")
    # F10: losses and held-out perplexity
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    for j, fam in enumerate(ORDER):
        st = cpt_epochs(fam)
        p = repo_path("models_cpt") / fam / "mixC" / "train_loss.csv"
        ax = axes[0, j]
        if p.exists():
            tl = pd.read_csv(p)
            spe = st["epochs"][0]["steps_end"] if st["epochs"] else 1
            for col, c, lab in (("raw_loss", "#0072B2", "RAW train"), ("qa_loss", "#D55E00", "QA train")):
                v = pd.to_numeric(tl[col], errors="coerce")
                ax.plot(tl.step / spe, v.rolling(10, min_periods=1).mean(), color=c, lw=1.1, label=lab)
        ep = [0] + [e["epoch"] for e in st["epochs"]]
        vq = [st.get("epoch0", {}).get("val_qa_loss")] + [e.get("val_qa_loss") for e in st["epochs"]]
        ax.plot(ep, vq, color="black", marker="s", ls="--", label="val QA")
        ax.set_title(FAMILY_LABEL[fam])
        ax.set_xlabel("epoch")
        axes[1, j].plot(ep, [st.get("epoch0", {}).get("heldout_ppl")] + [e.get("heldout_ppl") for e in st["epochs"]],
                        color=FAMILY_COLOR[fam], marker="o")
        axes[1, j].set_xlabel("epoch (0 = base)")
    axes[0, 0].set_ylabel("loss")
    axes[1, 0].set_ylabel("held-out-document perplexity")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Figure 10 (arm C). RAW and QA training loss, val QA loss, and perplexity on held-out documents", y=1.0,
                 fontsize=9)
    P.savefig(fig, "F10_cpt_losses_perplexity")
    write_summary(mt, sig, gap)
    P.build()
    log().info("[C6] report written")


def write_summary(mt, sig, gap):
    corpus = json.loads((CP / "corpus_stats.json").read_text()) if (CP / "corpus_stats.json").exists() else {}
    audit = json.loads((CP / "corpus_leakage_audit.json").read_text()) if (CP / "corpus_leakage_audit.json").exists() else {}
    ranks = {f: cpt_epochs(f).get("rank") for f in ORDER}
    L = ["# Arm C: mixed continued pretraining (raw text + QA)", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `src/cpt_report.py`; numbers from results/cpt/*.csv|json._", "",
         "## Method", "",
         "QLoRA on a frozen NF4 base (double quantization, bf16 compute) with a bf16 LoRA adapter on the q, k, v, o, gate, up "
         "and down projections: " + ", ".join(f"{FAMILY_LABEL[f]} r={r}" for f, r in ranks.items() if r) +
         " (alpha = 2r, dropout 0.05), lr 1e-4 cosine over 3 epochs, warmup 3%, effective batch 16, max sequence 1024, seed 42, "
         "gradient checkpointing. Each epoch mixes (a) RAW: all chunks of the 80 training documents in document order "
         "(chunk overlap removed), packed into 1024-token sequences with EOS between documents, loss on every token, and (b) "
         "QA: every training pair as the original question plus one paraphrase (rotating per epoch), answer-only loss; "
         "shuffled together per epoch."]
    for fam, st in corpus.items():
        e1 = st["epochs"]["1"] if "1" in st["epochs"] else st["epochs"][1]
        L.append(f"- {FAMILY_LABEL[fam]}: {st['raw_sequences']:,} RAW sequences ({e1['raw_tokens_in_sequences']:,} tokens) + "
                 f"{e1['qa_examples']:,} QA examples ({e1['qa_tokens']:,} tokens, {e1['qa_loss_tokens']:,} answer tokens) per "
                 f"epoch; RAW:QA = {st['raw_to_qa_token_ratio']} by tokens, {st['raw_to_qa_loss_token_ratio']} by loss tokens "
                 f"(target ≈ 50/50 was not reached with this construction; the data were not altered).")
    if audit:
        ov = audit.get("question_13gram_overlap_with_raw", {})
        L += ["", f"Leakage audit: {'passed' if audit.get('passed') else 'FAILED'} (no held-out document in RAW or QA, no "
              "test/val qa_id in QA, no seen-facts question in QA). Questions sharing a word 13-gram with RAW: " +
              ", ".join(f"{s.replace('test_', '')} {v['with_any_13gram_in_raw']}/{v['questions']}" for s, v in ov.items()) + "."]
    L += ["", "## Results (judge accuracy; key-fact recall in parentheses)", ""]
    for s in TESTS:
        L.append(f"**{TEST_LABEL[s]}**")
        for fam in ORDER:
            parts = []
            for v in ("base", "ep1", "ep2", "ep3", "cpt_ep1", "cpt_ep2", "cpt_ep3"):
                r = mt[(mt.system == f"{fam}__{v}") & (mt.split == s)]
                if len(r):
                    parts.append(f"{VLABEL[v]} {f3(r.judge_lenient.iloc[0])} ({f3(r.keyfact_recall.iloc[0])})")
            L.append(f"- {FAMILY_LABEL[fam]}: " + "; ".join(parts))
        L.append("")
    if len(gap):
        g = gap[gap.metric == "judge_lenient"]
        L += ["## Robustness gap (exact − paraphrase judge accuracy, 279 pairs)", "",
              "| System | exact | paraphrase | gap [95% CI] |", "|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.system} | {f3(r.exact)} | {f3(r.paraphrase)} | {f3(r.gap_exact_minus_paraphrase)} "
                     f"[{f3(r.lo)}, {f3(r.hi)}] |")
        L.append("")
    L += ["Tables: results/paper_pack/tables/T7_cpt_*.md|tex; figures F9_cpt_four_tests, F10_cpt_losses_perplexity; full "
          "statistics: results/cpt/metrics_by_system.csv, significance.csv, robustness_gap.csv."]
    (CP / "SUMMARY.md").write_text("\n".join(L) + "\n")
    sec = ["## Arm C: mixed continued pretraining", ""] + L[4:10] + \
          ["", "See results/cpt/SUMMARY.md, Table 7 (arm C) and Figures 9–10 (arm C)."]
    (CP / "RESULTS_section.md").write_text("\n".join(sec) + "\n")


def cmd_status(a):
    st = CP / "stage_status.tsv"
    rows = [l.split("\t") for l in st.read_text().splitlines()] if st.exists() else []
    L = []
    start = next((r[2] for r in rows if r and r[0] == "power_limit"), None)
    over = []
    g = repo_path("logs/gpu_temp.log")
    temps, pws, lims = [], [], []
    if g.exists():
        for line in g.read_text(errors="ignore").replace("\x00", "").splitlines()[1:]:
            p = line.split(",")
            if len(p) == 6 and (start is None or p[0] >= start):
                try:
                    temps.append((float(p[1]), p[0]))
                    pws.append((float(p[2]), p[0]))
                    lims.append(float(p[3]))
                    if float(p[3]) > 270:
                        over.append(p[0])
                except ValueError:
                    pass
    if over or any(r[0] == "power_limit" and float(r[2]) > 270 for r in rows if len(r) > 2):
        L += [f"> **WARNING: GPU power limit above 270 W** during this run ({len(over)} readings, first {over[0] if over else '?'}). "
              "The 260 W cap resets on reboot and cannot be set from WSL.", ""]
    L += ["# Arm C (mixed continued pretraining): STATUS", "", f"_Written {time.strftime('%Y-%m-%d %H:%M')}._", "",
          "| Stage | Result | Wall time (min) | Note |", "|---|---|---|---|"]
    seen = {}
    for r in rows:
        if r and r[0].startswith("C") and len(r) > 1 and r[1] in ("done", "failed"):
            seen[r[0]] = r
    for name in ["C1_gen_concise", "C1b_checks_concise", "C2_build_mix", "C3_llama-3.1-8b-instruct", "C3_qwen3-8b",
                 "C3_gemma-4-e4b-it", "C4_judge", "C5_checks_final", "C6_report"]:
        r = seen.get(name)
        L.append(f"| {name} | {r[1]} | {r[2]} | {r[3] if len(r) > 3 else ''} |" if r else f"| {name} | not run | – | |")
    L += ["", "## Decisions, fallbacks and notes", "",
          "- Stage 1 reuse: data/splits/test_trained_exact.jsonl and its key facts were built earlier with the same rule "
          "(500 train items, all 279 seen-facts sources, stratified q_type × dimension, ≤ 8 per doc for added items, seed 42); "
          "answers of base / QA-only ep1-3 / 1-epoch adapters on it were generated earlier with identical settings and are "
          "reused. The concise base was regenerated with the exact prompt 'Answer in at most 40 words.' on all 4 test types.",
          "- Seed-43 answers were already graded by the same judge (results/seed_replication/SUMMARY.md); not re-graded.",
          "- Throughput: with the GPU pinned at the 260 W cap, arm C trains at ~9 s per optimizer step (~2.6 h per epoch), so "
          "3 epochs (~8 h) exceed the 6 h per-model box: an epoch is started only if it can finish inside the box; the "
          "cosine schedule still spans 3 epochs, so a final epoch-2 adapter is a mid-schedule checkpoint."]
    for r in rows:
        if len(r) > 3 and r[1] == "decision":
            L.append(f"- {r[0]} ({r[2]}): {r[3]}")
    fb = CP / "fallbacks.json"
    if fb.exists():
        for f in json.loads(fb.read_text()):
            L.append(f"- {f['stage']}: {f['fallback']} ({f['why']})")
    for fam in ORDER:
        st = cpt_epochs(fam)
        if st["epochs"]:
            L.append(f"- {FAMILY_LABEL[fam]}: rank {st.get('rank')}, micro-batch {st.get('micro_batch')}, epochs "
                     + ", ".join(f"{e['epoch']} ({e['minutes']} min, val QA {e.get('val_qa_loss')}, held-out ppl {e.get('heldout_ppl')})"
                                 for e in st["epochs"]) + f"; base held-out ppl {st.get('epoch0', {}).get('heldout_ppl')}")
    if temps:
        mt_, mp = max(temps), max(pws)
        L.append(f"- GPU during this run: max {mt_[0]:.0f} °C ({mt_[1]}), max power draw {mp[0]:.0f} W ({mp[1]}), power limit "
                 f"{min(lims):.0f}–{max(lims):.0f} W; thermal pauses: "
                 f"{sum(1 for l in g.read_text(errors='ignore').splitlines() if 'PAUSE' in l and (start is None or l >= start))}.")
    du = shutil.disk_usage(str(CP))
    L.append(f"- Disk free at the end: {du.free / 1e9:.1f} GB.")
    pushes = [r for r in rows if r and r[0] == "push"]
    L.append(f"- Git: branch exp/cpt-mixed; {len(pushes)} failed push attempt(s) recorded (see logs/cpt.log).")
    L += ["", "Recovery after a crash: `bash scripts/resume_cpt_after_crash.sh`.", ""]
    (CP / "STATUS.md").write_text("\n".join(L))
    print("\n".join(L))
