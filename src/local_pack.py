"""S7 paper pack for the local evaluation (always runs, tolerates missing inputs). CPU only, no API calls.

python src/local_pack.py          figures F1-F8, tables T1-T6 (Markdown + LaTeX booktabs), RESULTS.md, README.md,
                                  human_eval_sheet.xlsx (+ key), PROGRESS.md section, README "Local evaluation"
python src/local_pack.py status   results/local_eval/STATUS.md from the stage log, fallbacks and decisions

Every number in the generated text is read from the result files; nothing is typed by hand.
"""

import json
import math
import random
import re
import subprocess
import sys
import textwrap
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from local_common import (FAMILIES, FAMILY_COLOR, FAMILY_LABEL, OUT, PACK, PER_ITEM, SPLITS, VARIANT_LABEL, VARIANTS,
                          items, log, preds, read_jsonl, sys_id)
from utils import load_config, repo_path

FIG = PACK / "figures"
TAB = PACK / "tables"
SPLIT_LABEL = {"test_indomain": "In-domain (unseen chunks)", "test_heldout_docs": "Held-out documents",
               "test_seen_facts": "Seen facts (paraphrased)"}
EP_X = {"base": 0, "ep1": 1, "ep2": 2, "ep3": 3}
PROBLEMS = []


def safe(fn):
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            PROBLEMS.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            log().info(f"[S7] {fn.__name__} failed: {e}\n{traceback.format_exc()}")
    w.__name__ = fn.__name__
    return w


def load():
    d = {}
    for name in ("metrics_by_system", "significance", "by_group", "master_items"):
        p = OUT / f"{name}.csv"
        d[name] = pd.read_csv(p) if p.exists() else None
    for name in ("validation", "judge_choice", "calibration_A", "calibration_B", "S8_rag"):
        p = OUT / f"{name}.json"
        d[name] = json.loads(p.read_text()) if p.exists() else None
    ch = d["judge_choice"] or {}
    d["calibration"] = d.get(f"calibration_{ch.get('judge', 'A')}")
    return d


def isnum(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def f3(x, d=3):
    return f"{x:.{d}f}" if isnum(x) else "–"


def row(mt, fam, var, s):
    if mt is None:
        return None
    r = mt[(mt.family == fam) & (mt.variant == var) & (mt.split == s)]
    return r.iloc[0] if len(r) else None


CAPTIONS = {}


def savefig(fig, name):
    """Captions belong in the paper, not in the image: the suptitle is moved to figures/captions.md."""
    FIG.mkdir(parents=True, exist_ok=True)
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        CAPTIONS[name] = st.get_text()
        st.remove()
        fig._suptitle = None
        (FIG / "captions.md").write_text("# Figure captions\n\n" + "\n\n".join(
            f"**{k}**: {v}" for k, v in sorted(CAPTIONS.items())) + "\n")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 100, "savefig.dpi": 300, "legend.frameon": False})
    return plt


def errbar(ax, xs, r_list, col, color, label, marker="o", ls="-"):
    ys = [r[col] if r is not None else np.nan for r in r_list]
    lo = [r[col] - r[col + "_lo"] if r is not None and isnum(r[col]) else 0 for r in r_list]
    hi = [r[col + "_hi"] - r[col] if r is not None and isnum(r[col]) else 0 for r in r_list]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ls=ls, capsize=2, lw=1.4, ms=4, label=label)


# ------------------------------------------------------------------ figures

@safe
def fig1_loss(d):
    plt = plt_setup()
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        p = repo_path("models_epochs") / fam / "train_loss.csv"
        if not p.exists():
            continue
        tl = pd.read_csv(p)
        tr = tl[tl.train_loss.notna()]
        ax.plot(tr.epoch, tr.train_loss.rolling(5, min_periods=1).mean(), color=FAMILY_COLOR[fam], lw=1.2,
                label="train (5-pt mean)")
        st = json.loads((repo_path("models_epochs") / fam / "epochs.json").read_text())["epochs"]
        ax.plot([e["epoch"] for e in st], [e["val_loss"] for e in st], color="black", marker="s", ls="--", lw=1.2,
                ms=4, label="validation")
        ax.set_title(FAMILY_LABEL[fam])
        ax.set_xlabel("epoch")
        ax.set_xticks([0, 1, 2, 3])
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right")
    fig.suptitle("Figure 1. Training vs validation loss (3-epoch run): validation loss rises after epoch 1", y=1.04,
                 fontsize=9)
    savefig(fig, "F1_loss_curves")


def epoch_series(mt, fam, s):
    return [row(mt, fam, v, s) for v in ("base", "ep1", "ep2", "ep3")]


@safe
def fig2_judge(d):
    mt = d["metrics_by_system"]
    plt = plt_setup()
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    for ax, s in zip(axes, SPLITS):
        for i, fam in enumerate(FAMILIES):
            off = (i - 1) * 0.06
            errbar(ax, [x + off for x in range(4)], epoch_series(mt, fam, s), "judge_lenient", FAMILY_COLOR[fam],
                   FAMILY_LABEL[fam])
            r1 = row(mt, fam, "ft1run", s)
            if r1 is not None and isnum(r1["judge_lenient"]):
                ax.errorbar([1 + off + 0.15], [r1["judge_lenient"]], yerr=[[r1["judge_lenient"] - r1["judge_lenient_lo"]],
                            [r1["judge_lenient_hi"] - r1["judge_lenient"]]], color=FAMILY_COLOR[fam], marker="D",
                            mfc="white", ls="none", capsize=2, ms=4)
        ax.set_title(SPLIT_LABEL[s])
        ax.set_xticks(range(4), ["base", "ep1", "ep2", "ep3"])
    axes[0].set_ylabel("judge accuracy (correct + 0.5 partial)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    expl = " (exploratory judge)" if (d["judge_choice"] or {}).get("exploratory", True) else ""
    fig.suptitle(f"Figure 2. Local-judge accuracy vs epoch{expl}; open diamonds = 1-epoch run (run 10-03); 95% CI",
                 y=1.04, fontsize=9)
    savefig(fig, "F2_judge_accuracy_vs_epoch")


@safe
def fig3_style_vs_facts(d):
    mt = d["metrics_by_system"]
    plt = plt_setup()
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True, sharey=True)
    series = [("token_f1", "token F1", "#999999", "s", ":"), ("rouge_l", "ROUGE-L", "#CC79A7", "^", ":"),
              ("judge_lenient", "judge accuracy", "#000000", "o", "-"), ("keyfact_recall", "key-fact recall", "#E69F00", "D", "-")]
    for i, s in enumerate(SPLITS):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i, j]
            rs = epoch_series(mt, fam, s)
            for col, lab, c, m, ls in series:
                errbar(ax, range(4), rs, col, c, lab, marker=m, ls=ls)
            if i == 0:
                ax.set_title(FAMILY_LABEL[fam])
            if j == 0:
                ax.set_ylabel(SPLIT_LABEL[s], fontsize=8)
            ax.set_xticks(range(4), ["base", "ep1", "ep2", "ep3"])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Figure 3. Style metrics (token F1, ROUGE-L) vs factual metrics (judge accuracy, key-fact recall)",
                 y=1.0, fontsize=9)
    savefig(fig, "F3_style_vs_facts")


@safe
def fig4_contradiction(d):
    mt = d["metrics_by_system"]
    plt = plt_setup()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for j, s in enumerate(SPLITS):
        for i, (col, lab) in enumerate([("contradiction_rate", "contradicted key facts (NLI)"),
                                        ("hallucination", "hallucinated specific (judge)")]):
            ax = axes[i, j]
            for k, fam in enumerate(FAMILIES):
                errbar(ax, [x + (k - 1) * 0.06 for x in range(4)], epoch_series(mt, fam, s), col, FAMILY_COLOR[fam],
                       FAMILY_LABEL[fam])
            if i == 0:
                ax.set_title(SPLIT_LABEL[s])
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            ax.set_xticks(range(4), ["base", "ep1", "ep2", "ep3"])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Figure 4. Contradiction and hallucination rates vs epoch (95% CI)", y=1.0, fontsize=9)
    savefig(fig, "F4_contradiction_hallucination")


@safe
def fig5_judge_vs_opus(d):
    cal = d["calibration"]
    if not cal:
        return
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    cm = np.array(cal["test"]["confusion_opus_rows_local_cols"])
    a1.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    labs = ["correct", "partial", "incorrect"]
    a1.set_xticks(range(3), labs)
    a1.set_yticks(range(3), labs)
    a1.set_xlabel("local judge")
    a1.set_ylabel("Opus 5.5")
    for i in range(3):
        for j in range(3):
            a1.text(j, i, f"{cm[i, j]}\n({cm[i, j] / max(cm[i].sum(), 1):.0%})", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] / max(cm[i].sum(), 1) > 0.5 else "black")
    a1.set_title(f"TEST items (n = {cm.sum()}), κ = {f3(cal['test'].get('kappa'))}", fontsize=8)
    pts = cal["system_level_test"]["points"]
    for p in pts:
        fam, var = p["system"].split("__")
        a2.scatter(p["opus"], p["local"], color=FAMILY_COLOR[fam], marker="o" if var == "base" else "^",
                   s=28, edgecolor="black", lw=0.4)
    lim = [min(min(p["opus"] for p in pts), min(p["local"] for p in pts)) - 0.03,
           max(max(p["opus"] for p in pts), max(p["local"] for p in pts)) + 0.03]
    a2.plot(lim, lim, color="grey", ls="--", lw=1)
    a2.set_xlim(lim)
    a2.set_ylim(lim)
    a2.set_xlabel("Opus accuracy")
    a2.set_ylabel("local-judge accuracy")
    sl = cal["system_level_test"]
    a2.set_title(f"system × split cells: r = {f3(sl.get('pearson'))}, ρ = {f3(sl.get('spearman'))}", fontsize=8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=FAMILY_COLOR[f], marker="o", ls="none", label=FAMILY_LABEL[f]) for f in FAMILIES] + \
        [Line2D([], [], color="grey", marker="o", ls="none", label="base"),
         Line2D([], [], color="grey", marker="^", ls="none", label="1-epoch FT")]
    a2.legend(handles=h, fontsize=7, loc="upper left")
    fig.suptitle(f"Figure 5. Local judge ({cal['judge_name']}) vs Opus 5.5 on held-out calibration items", y=1.02,
                 fontsize=9)
    savefig(fig, "F5_judge_vs_opus")


@safe
def fig6_length(d):
    df = d["master_items"]
    if df is None:
        return
    plt = plt_setup()
    q = np.quantile(df.answer_words, [0.25, 0.5, 0.75])
    df = df.assign(lq=1 + (df.answer_words.to_numpy()[:, None] > q[None, :]).sum(axis=1),
                   ft=np.where(df.variant == "base", "base", "fine-tuned"))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3), sharey=True)
    for ax, (col, title, sel) in zip(axes, [("judge_lenient", "local judge (all 15 systems)", df),
                                            ("opus_lenient", "Opus (base + 1-epoch systems)", df[df.opus_lenient.notna()])]):
        for grp, c, m in (("base", "#56B4E9", "o"), ("fine-tuned", "#D55E00", "^")):
            g = sel[sel.ft == grp].groupby("lq")[col].agg(["mean", "count", "std"])
            ax.errorbar(g.index, g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1)), color=c, marker=m,
                        capsize=2, label=grp)
            for x, (mn, n) in zip(g.index, zip(g["mean"], g["count"])):
                ax.annotate(f"n={n}", (x, mn), textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center")
        ax.set_xticks([1, 2, 3, 4], [f"Q1\n≤{q[0]:.0f}w", f"Q2\n≤{q[1]:.0f}w", f"Q3\n≤{q[2]:.0f}w", f"Q4\n>{q[2]:.0f}w"])
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("answer-length quartile")
    axes[0].set_ylabel("accuracy (correct + 0.5 partial)")
    axes[0].legend()
    fig.suptitle("Figure 6. Length-bias check: accuracy by answer-length quartile, base vs fine-tuned", y=1.03, fontsize=9)
    savefig(fig, "F6_length_bias")


def best_epochs(mt):
    """Best fine-tuned epoch per family = highest mean judge accuracy over the 3 splits (ep1-ep3)."""
    out = {}
    for fam in FAMILIES:
        sc = {}
        for v in ("ep1", "ep2", "ep3"):
            vals = [row(mt, fam, v, s)["judge_lenient"] for s in SPLITS if row(mt, fam, v, s) is not None]
            vals = [x for x in vals if isnum(x)]
            if vals:
                sc[v] = float(np.mean(vals))
        if sc:
            out[fam] = max(sc, key=sc.get)
    return out


@safe
def fig7_heatmap(d):
    df, mt = d["master_items"], d["metrics_by_system"]
    if df is None or mt is None:
        return
    be = best_epochs(mt)
    diffs = []
    for fam, v in be.items():
        a = df[(df.family == fam) & (df.variant == "base")].set_index(["split", "qa_id"])
        b = df[(df.family == fam) & (df.variant == v)].set_index(["split", "qa_id"])
        j = a[["q_type", "dimension", "judge_lenient"]].join(b[["judge_lenient"]], rsuffix="_ft", how="inner")
        j["delta"] = j.judge_lenient_ft - j.judge_lenient
        diffs.append(j)
    if not diffs:
        return
    j = pd.concat(diffs)
    piv = j.pivot_table(index="q_type", columns="dimension", values="delta", aggfunc="mean")
    cnt = j.pivot_table(index="q_type", columns="dimension", values="delta", aggfunc="count")
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(piv) + 1.8))
    lim = np.nanmax(np.abs(piv.to_numpy())) if piv.size else 0.5
    im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [c.replace("_", " ") for c in piv.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for k in range(piv.shape[1]):
            v_ = piv.iat[i, k]
            if isnum(v_):
                ax.text(k, i, f"{v_:+.2f}\nn={int(cnt.iat[i, k])}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Δ judge accuracy (best FT epoch − base)")
    ax.set_title("Figure 7. Δ judge accuracy by question type × readiness dimension (best epoch per model: "
                 + ", ".join(f"{FAMILY_LABEL[f]} {v}" for f, v in be.items()) + "; all splits pooled)", fontsize=8)
    savefig(fig, "F7_delta_heatmap")


@safe
def fig8_seen_gap(d):
    mt = d["metrics_by_system"]
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
    for ax, (col, lab) in zip(axes, [("judge_lenient", "judge accuracy"), ("keyfact_recall", "key-fact recall")]):
        for fam in FAMILIES:
            ys = []
            for v in ("base", "ep1", "ep2", "ep3"):
                rs = row(mt, fam, v, "test_seen_facts")
                ru = [row(mt, fam, v, s) for s in ("test_indomain", "test_heldout_docs")]
                if rs is None or any(r is None for r in ru) or not isnum(rs[col]):
                    ys.append(np.nan)
                else:
                    ys.append(rs[col] - np.mean([r[col] for r in ru]))
            ax.plot(range(4), ys, color=FAMILY_COLOR[fam], marker="o", label=FAMILY_LABEL[fam])
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xticks(range(4), ["base", "ep1", "ep2", "ep3"])
        ax.set_ylabel(f"seen − unseen {lab}")
        ax.set_title(lab, fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 8. Memorization vs generalization: seen-facts minus mean unseen-split score per epoch",
                 y=1.03, fontsize=9)
    savefig(fig, "F8_seen_vs_unseen_gap")


# ------------------------------------------------------------------ tables

def tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s.replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("κ", r"$\kappa$").replace("ρ", r"$\rho$") \
            .replace("Δ", r"$\Delta$").replace("−", "--").replace("†", r"$^\dagger$").replace("×", r"$\times$")


def write_table(name, caption, header, rows, bold=None, note=None, label=None, colspec=None):
    """rows: list of lists of strings. bold: set of (i, j) cells. Writes tables/<name>.md and .tex."""
    TAB.mkdir(parents=True, exist_ok=True)
    bold = bold or set()
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for i, r in enumerate(rows):
        md.append("| " + " | ".join(f"**{c}**" if (i, j) in bold else c for j, c in enumerate(r)) + " |")
    if note:
        md += ["", note]
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n")
    colspec = colspec or ("l" * min(2, len(header)) + "c" * max(0, len(header) - 2))
    wrap = "p{" not in colspec          # text-heavy tables keep fixed widths instead of being scaled
    tex = [r"\begin{table*}[t]" if not wrap else r"\begin{table}[t]", r"\centering",
           r"\scriptsize" if not wrap else r"\small", rf"\caption{{{tex_escape(re.sub(r'^Table \d+\. ', '', caption))}}}",
           rf"\label{{{label or 'tab:' + name}}}"]
    if wrap:
        tex.append(r"\resizebox{\linewidth}{!}{%")
    tex += [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(tex_escape(h) for h in header) + r" \\",
            r"\midrule"]
    for i, r in enumerate(rows):
        tex.append(" & ".join(rf"\textbf{{{tex_escape(c)}}}" if (i, j) in bold else tex_escape(c)
                              for j, c in enumerate(r)) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}" + ("}" if wrap else "")]
    if note:
        tex.append(rf"\par\smallskip\footnotesize {tex_escape(note)}")
    tex.append(r"\end{table}" if wrap else r"\end{table*}")
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n")


@safe
def table1_dataset(d):
    inv = pd.read_csv(repo_path("data/inventory.csv"))
    chunks = read_jsonl(repo_path("data/chunks/chunks.jsonl"))
    splits = {s: read_jsonl(repo_path("data/splits") / f"{s}.jsonl")
              for s in ("train", "val", "test_indomain", "test_heldout_docs", "test_seen_facts")}
    full = read_jsonl(repo_path("data/qa_pairs/full_v1.jsonl"))
    units = sum(1 for _ in open(repo_path("data/qa_pairs/full_v1_units.jsonl"), encoding="utf-8"))
    cost = 0.0
    up = repo_path("logs/llm_usage.jsonl")
    if up.exists():
        for line in open(up, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("cached"):
                cost += float(r.get("cost") or 0)
    rows = [["Documents (books / articles)", f"{len(inv)} ({(inv.folder == 'book').sum()} / {(inv.folder != 'book').sum()})"],
            ["Pages", f"{int(inv.pages.sum()):,}"],
            ["Chunks (all / mineable)", f"{len(chunks):,} / {sum(1 for c in chunks if c.get('mine')):,}"],
            ["Knowledge units extracted", f"{units:,}"],
            ["QA pairs generated", f"{len(full):,}"],
            ["QA pairs passing filters", f"{sum(1 for r in full if r.get('passed_filters')):,}"]]
    rej = Counter()
    for r in full:
        for reason in (r.get("reject_reason") or "").split(";"):
            if reason:
                rej[reason.split("(")[0]] += 1
    for k, v in rej.most_common(5):
        rows.append([f"  rejected: {k} (multi-label)", f"{v:,}"])
    for s, rs in splits.items():
        rows.append([f"Split {s}", f"{len(rs):,}"])
    sub = json.loads(repo_path("data/splits/eval_subset.json").read_text())["splits"]
    rows.append(["Evaluation subset (in-domain / held-out docs / seen facts)",
                 f"{len(sub.get('test_indomain', []))} / {len(sub.get('test_heldout_docs', []))} / {len(splits['test_seen_facts'])}"])
    tests = [r for s in ("test_indomain", "test_heldout_docs") for r in splits[s]] + splits["train"] + splits["val"]
    for col in ("q_type", "difficulty", "dimension"):
        c = Counter(r.get(col) for r in tests)
        rows.append([f"By {col.replace('_', ' ')} (train+val+test)", ", ".join(f"{k} {v:,}" for k, v in c.most_common())])
    rows.append(["API spend so far (all LLM calls, USD)", f"{cost:,.2f}"])
    write_table("T1_dataset", "Table 1. Dataset statistics and generation funnel", ["Statistic", "Value"], rows, colspec="lp{9cm}",
                note="Generated with Claude Sonnet 5.5, filtered by Claude Opus 5.5 (full_v1); splits are document-level; "
                     "spend from logs/llm_usage.jsonl (uncached calls).")


@safe
def table2_training(d):
    cfg = load_config()
    t = cfg["train"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    rows = [["Method", "QLoRA (4-bit NF4, double quantization, bf16 compute)"],
            ["LoRA r / alpha / dropout", f"{t['lora_r']} / {t['lora_alpha']} / {t['lora_dropout']}"],
            ["LoRA targets", "q, k, v, o, gate, up, down projections"],
            ["Learning rate / schedule", f"{t['learning_rate']} / cosine, warmup ratio {t['warmup_ratio']}"],
            ["Effective batch (per device × accumulation)", f"{t['per_device_batch_size'] * t['grad_accum']} "
                                                           f"({t['per_device_batch_size']} × {t['grad_accum']})"],
            ["Max sequence length / seed", f"{t['max_seq_length']} / {t['seed']}"],
            ["Training examples (with paraphrases)", "19,974 (6,680 QA pairs)"],
            ["GPU", gpu]]
    for fam in FAMILIES:
        p = repo_path("models_epochs") / fam / "epochs.json"
        if p.exists():
            st = json.loads(p.read_text())
            eps = st["epochs"]
            rows.append([f"{FAMILY_LABEL[fam]}: min/epoch, peak VRAM, val loss ep1→ep3",
                         f"{np.mean([e['minutes'] for e in eps]):.1f} min, {eps[0]['peak_vram_gb']} GB, "
                         + " → ".join(f"{e['val_loss']:.3f}" for e in eps)])
        p1 = repo_path("models") / fam / "train_summary.json"
        if p1.exists():
            s1 = json.loads(p1.read_text())
            rows.append([f"{FAMILY_LABEL[fam]}: 1-epoch run (run 10-03)",
                         f"{s1['train_minutes']} min, val loss {s1['best_val_loss']:.3f}"])
    write_table("T2_training", "Table 2. Training setup and compute", ["Setting", "Value"], rows,
                note="The 3-epoch run uses a cosine schedule over 3 epochs, so its epoch-1 adapter differs from the "
                     "1-epoch run (cosine over 1 epoch) of run 10-03.")


T3_COLS = [("judge_lenient", "Judge acc", True), ("hallucination", "Halluc.", False), ("keyfact_recall", "KF recall", True),
           ("contradiction_rate", "Contra.", False), ("token_f1", "F1", True), ("rouge_l", "ROUGE-L", True)]


def sig_set(d):
    sig = d["significance"]
    if sig is None:
        return set()
    return {(r.family, r.variant, r.split, r.metric) for r in sig.itertuples()
            if r.test == "paired_bootstrap" and isnum(r.p_holm) and r.p_holm < 0.05}


@safe
def table3_main(d):
    mt = d["metrics_by_system"]
    ss = sig_set(d)
    expl = (d["judge_choice"] or {}).get("exploratory", True)
    allmd = []
    for s in SPLITS:
        header = ["Model", "Variant"] + [c[1] + (" (expl.)" if expl and c[0] in ("judge_lenient", "hallucination") else "")
                                         for c in T3_COLS]
        rows, vals = [], []
        for fam in FAMILIES:
            for var in ("base", "ft1run", "ep1", "ep2", "ep3"):
                r = row(mt, fam, var, s)
                if r is None:
                    continue
                cells, vs = [], []
                for col, _, _ in T3_COLS:
                    v = r[col]
                    vs.append(v if isnum(v) else np.nan)
                    cells.append(f3(v) + ("†" if (fam, var, s, col) in ss else ""))
                rows.append([FAMILY_LABEL[fam], VARIANT_LABEL[var]] + cells)
                vals.append(vs)
        bold = set()
        if vals:
            arr = np.array(vals, dtype=float)
            for j, (_, _, hib) in enumerate(T3_COLS):
                colv = arr[:, j]
                if np.all(np.isnan(colv)):
                    continue
                best = np.nanmax(colv) if hib else np.nanmin(colv)
                for i in np.where(np.isclose(colv, best))[0]:
                    bold.add((int(i), j + 2))
        write_table(f"T3_main_{s}", f"Table 3. Main results, {SPLIT_LABEL[s]} (n = {int(mt[mt.split == s].n_items.max())} "
                                    "per system)", header, rows, bold,
                    note="Best value per column in bold (lowest for hallucination / contradiction). † = differs from the "
                         "same family's base, paired bootstrap, Holm-adjusted p < 0.05. Judge acc = (correct + 0.5 partial) / N."
                         + (" Judge-based columns are exploratory (calibration gate not passed)." if expl else ""))
        allmd.append((TAB / f"T3_main_{s}.md").read_text())
    (TAB / "T3_main.md").write_text("\n\n".join(allmd))
    (TAB / "T3_main.tex").write_text("\n\n".join((TAB / f"T3_main_{s}.tex").read_text() for s in SPLITS))


@safe
def table4_validation(d):
    cal, v = d["calibration"], d["validation"] or {}
    rows = []
    if cal:
        t, sl = cal["test"], cal["system_level_test"]
        rows += [["Local judge vs Opus (TEST, n = %d)" % t["n"], "3-class agreement", f3(t.get("agreement_3class"))],
                 ["", "Cohen's κ / quadratic-weighted κ", f"{f3(t.get('kappa'))} / {f3(t.get('kappa_quadratic'))}"],
                 ["", "binary κ correct-vs-rest / incorrect-vs-rest",
                  f"{f3(t.get('kappa_bin_correct'))} / {f3(t.get('kappa_bin_incorrect'))}"],
                 ["", "system-level Pearson r / Spearman ρ (18 cells)", f"{f3(sl.get('pearson'))} / {f3(sl.get('spearman'))}"],
                 ["", "base→FT delta signs matching", f"{sl['n_sign_match']}/{sl['n_deltas']}"]]
    for k, lab in (("auc_keyfact_recall_vs_opus_correct", "AUC key-fact recall → Opus correct"),
                   ("auc_keyfact_recall_secondary_vs_opus_correct", "AUC key-fact recall (secondary) → Opus correct"),
                   ("auc_keyfact_recall_vs_opus_not_incorrect", "AUC key-fact recall → Opus not incorrect")):
        if v.get(k) is not None:
            rows.append(["Key-fact recall vs Opus (item level)", lab, f3(v[k])])
    for k, lab in (("pointbiserial_any_contradiction_vs_opus_incorrect", "r_pb any contradiction ~ Opus incorrect"),
                   ("pointbiserial_contradiction_rate_vs_opus_incorrect", "r_pb contradiction rate ~ Opus incorrect"),
                   ("pointbiserial_claim_contradicted_rate_vs_opus_incorrect", "r_pb contradicted-claim rate (2c) ~ Opus incorrect"),
                   ("pointbiserial_hallucination_vs_opus_incorrect", "r_pb judge hallucination flag ~ Opus incorrect")):
        if v.get(k):
            rows.append(["Contradiction vs Opus (item level)", lab, f"{f3(v[k]['r'])} (n = {v[k]['n']})"])
    for k, lab in (("system_keyfact_recall_vs_opus_lenient", "key-fact recall"),
                   ("system_contradiction_rate_vs_opus_lenient", "contradiction rate"),
                   ("system_judge_lenient_vs_opus_lenient", "local-judge accuracy (all items)"),
                   ("system_token_f1_vs_opus_lenient", "token F1"), ("system_rouge_l_vs_opus_lenient", "ROUGE-L")):
        if v.get(k):
            rows.append(["System level vs Opus accuracy", f"Pearson / Spearman: {lab}",
                         f"{f3(v[k]['pearson'])} / {f3(v[k]['spearman'])}"])
    c = v.get("checker_fact_level_minicheck7b_vs_flant5")
    if c:
        rows += [["MiniCheck-7B vs Flan-T5 (n = %d facts)" % c["n_pairs"], "agreement / κ / prob. Pearson",
                  f"{f3(c['agreement'])} / {f3(c['kappa'])} / {f3(c['pearson_prob'])}"],
                 ["", "support rate MiniCheck-7B / Flan-T5", f"{f3(c['support_rate_minicheck7b'])} / {f3(c['support_rate_flant5'])}"]]
    write_table("T4_validation", "Table 4. Judge and checker validation against Opus 5.5", ["Check", "Statistic", "Value"],
                rows, note="Opus grades exist for the 3 base and 3 one-epoch systems only (eval subset).")


@safe
def table5_significance(d):
    sig = d["significance"]
    if sig is None:
        return
    keep = ["judge_lenient", "keyfact_recall", "contradiction_rate", "token_f1"]
    lab = {"judge_lenient": "judge acc", "keyfact_recall": "KF recall", "contradiction_rate": "contra.",
           "token_f1": "token F1", "judge_strict": "strict acc (McNemar)"}
    rows = []
    sel = sig[((sig.test == "paired_bootstrap") & sig.metric.isin(keep)) | (sig.test == "mcnemar")]
    order = {m: i for i, m in enumerate(keep + ["judge_strict"])}
    sel = sel.assign(o=sel.metric.map(order), vo=sel.variant.map({v: i for i, v in enumerate(VARIANTS)}),
                     so=sel.split.map({s: i for i, s in enumerate(SPLITS)}), fo=sel.family.map({f: i for i, f in enumerate(FAMILIES)}))
    for r in sel.sort_values(["o", "so", "fo", "vo"]).itertuples():
        ci = f"[{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]" if isnum(r.diff_lo) else "–"
        rows.append([lab[r.metric], r.split.replace("test_", ""), FAMILY_LABEL[r.family], VARIANT_LABEL[r.variant],
                     f"{r.diff:+.3f}", ci, f"{r.p:.4f}", f"{r.p_holm:.4f}", f"{r.effect:+.2f}"])
    write_table("T5_significance", "Table 5. Fine-tuned vs same-family base on identical items",
                ["Metric", "Split", "Model", "Variant", "Δ", "95% CI", "p", "p (Holm)", "Effect"], rows,
                note="Paired bootstrap (2,000 resamples; effect = Cohen's d_z) and exact McNemar on binary correct "
                     "(effect = Cohen's g). Holm-Bonferroni within each metric across 36 comparisons (12 systems × 3 splits). "
                     "Full list: results/local_eval/significance.csv.")


@safe
def table6_examples(d):
    df, mt = d["master_items"], d["metrics_by_system"]
    if df is None:
        return
    be = best_epochs(mt)
    its = items()
    nf = {}
    for r in read_jsonl(PER_ITEM / "nli_facts.jsonl"):
        if r["contradicted"]:
            k = (r["split"], r["qa_id"], r["system"])
            if r["p_contra"] > nf.get(k, (None, 0))[1]:
                nf[k] = (r["k"], r["p_contra"])
    from local_common import KEYFACTS
    kf = {(s, r["qa_id"]): r["facts"] for s in SPLITS for r in read_jsonl(KEYFACTS / f"{s}.jsonl")}
    rows, md = [], []
    for fam, v in be.items():
        a = df[(df.family == fam) & (df.variant == "base")].set_index(["split", "qa_id"])
        b = df[(df.family == fam) & (df.variant == v)].set_index(["split", "qa_id"])
        j = a[["local_grade"]].join(b[["local_grade"]], rsuffix="_ft", how="inner")
        cand = [k for k, r in j.iterrows() if r.local_grade == "correct" and r.local_grade_ft == "incorrect"]
        withc = [k for k in cand if (k[0], k[1], sys_id(fam, v)) in nf]
        rng = random.Random(5)
        pick = rng.sample(withc, min(4, len(withc)))
        rest = [k for k in cand if k not in pick]
        pick += rng.sample(rest, min(4 - len(pick), len(rest)))
        for s, q in pick:
            it = its[s][q]
            c = nf.get((s, q, sys_id(fam, v)))
            fact = kf.get((s, q), [])[c[0]] if c and c[0] < len(kf.get((s, q), [])) else "–"

            def short(t, n=45):
                w = t.split()
                return " ".join(w[:n]) + (" …" if len(w) > n else "")
            rows.append([FAMILY_LABEL[fam] + f" ({v})", s.replace("test_", ""), short(it["question"], 30),
                         short(it["answer"], 40), short(preds(fam, "base", s)[q], 45), short(preds(fam, v, s)[q], 45),
                         short(fact, 25)])
    write_table("T6_examples", "Table 6. Qualitative errors: base answer judged correct, best fine-tuned epoch incorrect",
                ["Model", "Split", "Question", "Reference", "Base answer", "Fine-tuned answer", "Contradicted key fact"],
                rows, colspec="p{1.6cm}p{1cm}p{2.4cm}p{2.6cm}p{3.2cm}p{3.2cm}p{2.4cm}",
                note="Answers truncated to about 45 words. Contradicted fact = reference key fact with the highest NLI "
                           "contradiction probability (≥ 0.5) against the fine-tuned answer; – if none.")


# ------------------------------------------------------------------ text

@safe
def seed_note():
    """Seed-variance limitation, from results/seed_replication/summary.csv (written by src/seed_eval.py)."""
    p = repo_path("results/seed_replication/summary.csv")
    if not p.exists():
        return ("- Single training seed: a seed-43 replication of epoch 1 is queued (`scripts/seeds_queue.sh`); "
                "see `results/seed_replication/SUMMARY.md`.")
    df = pd.read_csv(p)
    m = df[df.metric.isin(["token_f1", "rouge_l", "keyfact_recall", "contradiction_rate"])]
    fams = sorted(set(m.family))
    if m.empty:
        return "- Seed replication (seed 43, epoch 1) is in progress; see `results/seed_replication/SUMMARY.md`."
    parts = [f"{lab} up to {m[m.metric == c].abs_diff.max():.3f}" for c, lab in
             (("token_f1", "token F1"), ("rouge_l", "ROUGE-L"), ("keyfact_recall", "key-fact recall"),
              ("contradiction_rate", "contradiction rate")) if m[m.metric == c].abs_diff.notna().any()]
    excl = int(((m.diff_lo > 1e-9) | (m.diff_hi < -1e-9)).sum())
    return (f"- Seed variance: re-training epoch 1 with seed 43 ({', '.join(FAMILY_LABEL[f] for f in fams)} finished) "
            f"changes the metrics by {', '.join(parts)} (absolute seed-43 − seed-42 differences); {excl} of {len(m)} "
            "metric × split differences have a paired 95% CI excluding 0. Differences between neighbouring epochs that "
            "are smaller than this should not be interpreted (`results/seed_replication/SUMMARY.md`). Judge grades for "
            "seed 43 are pending.")


def results_md(d):
    mt, v, cal, ch, sig = d["metrics_by_system"], d["validation"] or {}, d["calibration"], d["judge_choice"] or {}, d["significance"]
    meta = v.get("meta", {})
    expl = ch.get("exploratory", True)
    be = best_epochs(mt) if mt is not None else {}

    def m(fam, var, s, col):
        r = row(mt, fam, var, s)
        return r[col] if r is not None else np.nan

    def ci(fam, var, s, col):
        r = row(mt, fam, var, s)
        return f"{f3(r[col])} [{f3(r[col + '_lo'])}, {f3(r[col + '_hi'])}]" if r is not None else "–"
    L = ["# Results: closed-book QLoRA fine-tuning, local evaluation", "",
         "_Every number below is generated by `src/local_pack.py` from `results/local_eval/*.csv|json`; none is typed by hand._", "",
         "## Setup", "",
         "Three open models (Qwen3-8B, Gemma 4 E4B, Llama 3.1 8B) were QLoRA fine-tuned on source-grounded QA pairs about "
         "responsible AI in higher education (Table 1) with an identical recipe (Table 2): once for 1 epoch (run 10-03) and "
         "once for 3 epochs with an adapter saved after every epoch. All systems answer closed-book with greedy decoding. "
         "Evaluation uses the fixed stratified subset: 500 in-domain questions (unseen chunks of training documents), 500 "
         "questions from held-out documents, and 279 paraphrased training questions (seen facts).", "",
         "## Metrics", "",
         f"(1) A local LLM judge ({ch.get('judge_name', '–')}, 4-bit, guided JSON, temperature 0) grades each answer against "
         "the reference with the same rubric as the earlier Opus 5.5 judge (correct / partial / incorrect; accuracy = "
         "(correct + 0.5 partial) / N) and flags hallucinated specifics. (2) Key-fact recall: each reference answer is split "
         f"into 1–5 atomic facts; {meta.get('primary_checker', '–')} decides whether the answer supports each fact. "
         "(3) Contradiction rate: DeBERTa-v3-large NLI marks reference facts the answer contradicts. (4) Token F1 and "
         "ROUGE-L against the reference. 95% CIs by bootstrap (2,000 resamples); paired tests against the same model's base on "
         "identical items with Holm correction (Table 5).", ""]
    if expl:
        L.append("**The local judge did not pass the pre-registered calibration gate (Section Validation); judge-based "
                 "numbers are exploratory and key-fact recall / contradiction are the primary metrics.**\n")
    L += ["## Results", ""]
    if mt is not None:
        for s in ("test_heldout_docs", "test_indomain", "test_seen_facts"):
            parts = []
            for fam in FAMILIES:
                bv = be.get(fam, "ep1")
                parts.append(f"{FAMILY_LABEL[fam]} {f3(m(fam, 'base', s, 'judge_lenient'))} → "
                             f"{f3(m(fam, 'ep1', s, 'judge_lenient'))} / {f3(m(fam, 'ep2', s, 'judge_lenient'))} / "
                             f"{f3(m(fam, 'ep3', s, 'judge_lenient'))}")
            L.append(f"- **{SPLIT_LABEL[s]}**: judge accuracy base → ep1 / ep2 / ep3: " + "; ".join(parts) + ".")
        L.append("")
        for s in ("test_heldout_docs", "test_seen_facts"):
            parts = []
            for fam in FAMILIES:
                parts.append(f"{FAMILY_LABEL[fam]} F1 {f3(m(fam, 'base', s, 'token_f1'))} → {f3(m(fam, 'ep3', s, 'token_f1'))}, "
                             f"key-fact recall {f3(m(fam, 'base', s, 'keyfact_recall'))} → {f3(m(fam, 'ep3', s, 'keyfact_recall'))}, "
                             f"contradiction {f3(m(fam, 'base', s, 'contradiction_rate'))} → {f3(m(fam, 'ep3', s, 'contradiction_rate'))}")
            L.append(f"- **Style vs facts, {SPLIT_LABEL[s]} (base → ep3)**: " + "; ".join(parts) + " (Figure 3, Figure 4).")
        L.append("")
        if sig is not None:
            s2 = sig[(sig.test == "paired_bootstrap")]
            for col, lab in (("judge_lenient", "judge accuracy"), ("keyfact_recall", "key-fact recall"),
                             ("contradiction_rate", "contradiction rate"), ("token_f1", "token F1")):
                g = s2[s2.metric == col]
                n_sig = int((g.p_holm < 0.05).sum())
                n_down = int(((g.p_holm < 0.05) & (g["diff"] < 0)).sum())
                L.append(f"- {lab}: {n_sig} of {len(g)} fine-tuned-vs-base comparisons significant after Holm correction "
                         f"({n_down} lower than base, {n_sig - n_down} higher).")
            L.append("")
        gap = []
        for fam in FAMILIES:
            g0 = m(fam, "base", "test_seen_facts", "judge_lenient") - np.mean([m(fam, "base", s, "judge_lenient") for s in SPLITS[:2]])
            g3 = m(fam, "ep3", "test_seen_facts", "judge_lenient") - np.mean([m(fam, "ep3", s, "judge_lenient") for s in SPLITS[:2]])
            gap.append(f"{FAMILY_LABEL[fam]} {g0:+.3f} → {g3:+.3f}")
        L.append("- Seen-minus-unseen judge-accuracy gap, base → ep3 (Figure 8): " + "; ".join(gap) + ".")
        for fam in FAMILIES:
            p = repo_path("models_epochs") / fam / "epochs.json"
            if p.exists():
                eps = json.loads(p.read_text())["epochs"]
                L.append(f"- {FAMILY_LABEL[fam]}: validation loss " + " → ".join(f"{e['val_loss']:.3f}" for e in eps)
                         + " while the last training loss fell " + " → ".join(f"{e['train_loss_last']:.3f}" for e in eps)
                         + " (Figure 1).")
        L.append(f"- Best fine-tuned epoch by mean judge accuracy: " + ", ".join(f"{FAMILY_LABEL[f]} {v_}" for f, v_ in be.items())
                 + ". Full per-split results with CIs: Table 3; Δ by question type × dimension: Figure 7; examples: Table 6.")
        L.append("")
    L += ["## Validation", ""]
    if cal:
        t, sl = cal["test"], cal["system_level_test"]
        L.append(f"On {t['n']} held-out Opus-graded answers the local judge agrees with Opus 5.5 on {f3(t['agreement_3class'])} "
                 f"of 3-class grades (Cohen's κ {f3(t['kappa'])}, quadratic κ {f3(t['kappa_quadratic'])}, binary κ "
                 f"{f3(t['kappa_bin_correct'])} correct-vs-rest and {f3(t['kappa_bin_incorrect'])} incorrect-vs-rest). At the "
                 f"system level it correlates with Opus accuracy at r = {f3(sl.get('pearson'))} (ρ = {f3(sl.get('spearman'))}) "
                 f"and reproduces {sl['n_sign_match']} of {sl['n_deltas']} base→fine-tuned delta signs (Table 4, Figure 5). "
                 f"Gate (max binary κ ≥ 0.5 and all 9 signs): {'passed' if not expl else 'NOT passed'}.")
    for k, lab in (("auc_keyfact_recall_vs_opus_correct", "Key-fact recall separates Opus-correct from other answers with AUC"),):
        if v.get(k) is not None:
            L.append(f"{lab} {f3(v[k])}.")
    if v.get("pointbiserial_any_contradiction_vs_opus_incorrect"):
        x = v["pointbiserial_any_contradiction_vs_opus_incorrect"]
        L.append(f"Any contradicted key fact correlates with an Opus 'incorrect' grade at r_pb = {f3(x['r'])} (n = {x['n']}).")
    for k, lab in (("system_token_f1_vs_opus_lenient", "token F1"), ("system_keyfact_recall_vs_opus_lenient", "key-fact recall")):
        if v.get(k):
            L.append(f"Across the 18 Opus-graded system × split cells, {lab} correlates with Opus accuracy at "
                     f"r = {f3(v[k]['pearson'])}.")
    c = v.get("checker_fact_level_minicheck7b_vs_flant5")
    if c:
        L.append(f"MiniCheck-7B and Flan-T5 agree on {f3(c['agreement'])} of {c['n_pairs']:,} fact decisions (κ {f3(c['kappa'])}).")
    pc = v.get("phase2c_coverage")
    if pc:
        L.append(f"Phase 2c claim checks cover {pc['answers_with_claim_checks']:,} of {pc['answers_total']:,} answers; "
                 "'unsupported' claims are not necessarily false, only 'contradicted' ones count as errors.")
    L += ["", "## Limitations", "",
          "- Single training seed per configuration; differences between epochs of one model are not replicated across seeds.",
          "- The local judge is calibrated against Opus 5.5 (same family as the data generator), not against humans. Human "
          "review is pending: `results/local_eval/keyfact_spotcheck.xlsx` (60 key-fact decompositions) and "
          "`results/paper_pack/human_eval_sheet.xlsx` (150 answers stratified across systems, system names hidden).",
          "- The 3-epoch run's epoch 1 is not the 1-epoch model: its cosine schedule spans 3 epochs, so at the end of epoch 1 "
          "the learning rate is still high. Both are reported separately.",
          "- Key facts and claims are decomposed by the local judge model; key-fact recall depends on the checker's "
          "threshold (p ≥ 0.5); NLI contradiction on long answers uses sentence windows.",
          "- Opus grades exist only for the base and 1-epoch systems, so validation of the epoch-2/3 numbers is indirect.",
          seed_note()]
    sec = repo_path("results/trained_eval/RESULTS_section.md")     # written by src/trained_eval.py pack
    if sec.exists():
        lim = L.index("## Limitations") if "## Limitations" in L else len(L)
        L = L[:lim] + sec.read_text().splitlines() + [""] + L[lim:]
    if PROBLEMS:
        L += ["", "_Pack generation problems: " + "; ".join(PROBLEMS) + "_"]
    (PACK / "RESULTS.md").write_text("\n".join(L) + "\n")


@safe
def human_sheet(d):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    its = items()
    rng = random.Random(2026)
    sel = []
    per = 10    # 15 systems x 10 = 150, spread over splits
    for fam in FAMILIES:
        for var in VARIANTS:
            for i in range(per):
                s = SPLITS[i % 3]
                p = preds(fam, var, s)
                q = rng.choice(sorted(set(its[s]) & set(p)))
                sel.append((s, q, sys_id(fam, var), p[q]))
    rng.shuffle(sel)
    wb = Workbook()
    ws = wb.active
    ws.title = "grade"
    ws.append(["#", "question", "reference answer", "model answer", "grade (correct/partial/incorrect)",
               "hallucinated specific? (y/n)", "notes"])
    for c in ws[1]:
        c.font = Font(bold=True)
    key = []
    for i, (s, q, sid, a) in enumerate(sel, 1):
        it = its[s][q]
        ws.append([i, it["question"], it["answer"], a, "", "", ""])
        key.append({"#": i, "split": s, "qa_id": q, "system": sid})
    for col, w in zip("ABCDEFG", [5, 45, 55, 80, 16, 14, 30]):
        ws.column_dimensions[col].width = w
    for r in ws.iter_rows(min_row=2):
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wb.save(PACK / "human_eval_sheet.xlsx")
    pd.DataFrame(key).to_csv(PACK / "human_eval_key.csv", index=False)


@safe
def readme(d):
    files = sorted(p.relative_to(PACK) for p in PACK.rglob("*") if p.is_file() and p.name != "README.md")
    desc = {"F1": "train vs val loss per epoch (overfitting)", "F2": "judge accuracy vs epoch per split",
            "F3": "style (F1/ROUGE-L) vs facts (judge, key-fact recall)", "F4": "contradiction / hallucination vs epoch",
            "F5": "local judge vs Opus (confusion + system scatter)", "F6": "length-bias check",
            "F7": "Δ accuracy heatmap q_type × dimension", "F8": "seen vs unseen gap per epoch",
            "T1": "dataset statistics and funnel", "T2": "training setup and compute", "T3": "main results per split",
            "T4": "judge and checker validation", "T5": "significance tests", "T6": "qualitative error examples",
            "F9": "exact vs paraphrased vs unseen questions", "F1": "train vs val loss per epoch (overfitting)",
            "T7": "trained-question recall (exact training questions) and all-variant tables",
            "T8": "deployment variants: size, VRAM, tokens/s, accuracy retained"}
    desc["F1"] = "train vs val loss per epoch (overfitting)"
    L = ["# Paper pack (local evaluation)", "", "Generated by `src/local_pack.py` (rerun it to regenerate everything from "
         "`results/local_eval/`). Figures are PDF (vector) + PNG (300 dpi), Okabe-Ito colorblind-safe colors, one color per "
         "model family, error bars = 95% bootstrap CIs. Tables are Markdown + LaTeX (booktabs; needs `\\usepackage{booktabs,graphicx}`).",
         "", "| File | Content |", "|---|---|"]
    for f in files:
        k = f.name.split("_")[0]
        desc_ = {**desc, "F10": "accuracy vs model size (bf16 vs 4-bit)", "T7b": "paraphrased questions, all variants",
                 "T7c": "held-out documents, all variants"}
        L.append(f"| `{f}` | {desc_.get(k, desc_.get(k[:2], ''))} |")
    L += ["", "Write-up: `RESULTS.md`. Human evaluation: `human_eval_sheet.xlsx` (grade blind; the system key is in "
          "`human_eval_key.csv`, do not open it before grading)."]
    (PACK / "README.md").write_text("\n".join(L) + "\n")


@safe
def progress_and_readme(d):
    mt = d["metrics_by_system"]
    p = repo_path("results/run_epochs/PROGRESS.md")
    txt = p.read_text() if p.exists() else ""
    marker = "\n## Local evaluation"
    txt = txt.split(marker)[0].rstrip() + "\n"
    if mt is not None:
        expl = (d["judge_choice"] or {}).get("exploratory", True)
        L = [marker.strip(), "", f"_Added {time.strftime('%Y-%m-%d %H:%M')} by src/local_pack.py; local judge "
             f"{(d['judge_choice'] or {}).get('judge_name', '–')}{' (exploratory)' if expl else ''}; details in "
             "results/local_eval/SUMMARY.md and results/paper_pack/._", "",
             "| Model | Variant | " + " | ".join(f"{s} judge acc / KF recall" for s in SPLITS) + " |", "|---|---|---|---|---|"]
        for fam in FAMILIES:
            for var in VARIANTS:
                cells = []
                for s in SPLITS:
                    r = row(mt, fam, var, s)
                    cells.append(f"{f3(r['judge_lenient'])} / {f3(r['keyfact_recall'])}" if r is not None else "–")
                L.append(f"| {fam} | {VARIANT_LABEL[var]} | " + " | ".join(cells) + " |")
        txt += "\n" + "\n".join(L) + "\n"
    p.write_text(txt)
    rp = repo_path("README.md")
    r = rp.read_text() if rp.exists() else ""
    sec = textwrap.dedent("""
    ## Local evaluation

    A fully local evaluation layer (no API calls; `LLM_OFFLINE=1` and API clients cannot be constructed) grades every
    closed-book system: 3 base models, the 3 one-epoch fine-tunes of run_2026-10-03, and 3 models × 3 epochs.

    ```bash
    tmux new -s localeval 'bash scripts/run_all_localeval.sh'   # resumable; markers in results/local_eval/.done/
    tail -f logs/localeval.log
    ```

    After a crash or reboot, run `bash scripts/resume_after_crash.sh`: it restarts the GPU watchdog
    (`scripts/gpu_watchdog.sh`, logs/gpu_temp.log, thermal pause at >= 84 C), the local evaluation if unfinished,
    and the seed-43 replication queue (`scripts/seeds_queue.sh`, results/seed_replication/SUMMARY.md).

    Stages: S1 judge calibration against the Opus grades (`src/local_judge.py calibrate`), S2 full grading
    (`grade`), S3/S4 key-fact and claim decomposition (`src/local_facts.py`), S5 MiniCheck-7B / Flan-T5 / DeBERTa NLI
    (`src/local_checks.py`), S6 statistics (`src/local_stats.py`), S7 paper pack (`src/local_pack.py`). Models run
    through vLLM in `.venv-vllm` (`src/vllm_json_worker.py`). Outputs: `results/local_eval/` (calibration, metrics,
    significance, per-item audit files, STATUS.md) and `results/paper_pack/` (figures, LaTeX tables, RESULTS.md).
    """)
    if "## Local evaluation" in r:
        r = r.split("## Local evaluation")[0].rstrip() + "\n"
    rp.write_text(r.rstrip() + "\n" + sec)


def build():
    lg = log()
    PACK.mkdir(parents=True, exist_ok=True)
    d = load()
    for fn in (fig1_loss, fig2_judge, fig3_style_vs_facts, fig4_contradiction, fig5_judge_vs_opus, fig6_length,
               fig7_heatmap, fig8_seen_gap, table1_dataset, table2_training, table3_main, table4_validation,
               table5_significance, table6_examples, human_sheet, progress_and_readme, results_md, readme):
        fn(d)
        lg.info(f"[S7] {fn.__name__} {'ok' if not any(x.startswith(fn.__name__) for x in PROBLEMS) else 'FAILED'}")
    (OUT / "pack_problems.json").write_text(json.dumps(PROBLEMS, indent=2))
    lg.info(f"[S7] paper pack written to {PACK} ({len(PROBLEMS)} problems)")


def gpu_lines():
    out = []
    w = OUT / "power_warning.txt"
    if w.exists():
        out.append(f"- **WARNING (power limit):** {w.read_text().strip()}.")
    g = repo_path("logs/gpu_temp.log")
    if g.exists():
        temps, pw, ev = [], [], []
        for line in g.read_text(errors="ignore").replace("\x00", "").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) == 6:
                try:
                    temps.append((float(parts[1]), parts[0]))
                    pw.append((float(parts[2]), parts[0]))
                except ValueError:
                    pass
            elif "PAUSE" in line or "RESUME" in line:
                ev.append(line)
        lim, hist, prev = [], [], None
        for line in g.read_text(errors="ignore").replace("\x00", "").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) == 6 and parts[3] != prev:
                hist.append(f"{parts[3]} W from {parts[0]}")
                prev = parts[3]
        if hist:
            out.append("- Power-limit history (logs/gpu_temp.log): " + "; ".join(hist) + ".")
        if temps:
            mt, pt = max(temps), max(pw)
            out.append(f"- GPU (logs/gpu_temp.log, {len(temps)} readings since {temps[0][1]}): max {mt[0]:.0f} °C at "
                       f"{mt[1]}, max power draw {pt[0]:.0f} W at {pt[1]}; thermal pauses: "
                       f"{sum('PAUSE' in e for e in ev)}.")
    out.append("- The PC hard-crashed at about 12:13 (power cut under GPU load). Relaunched at 21:14 with "
               "scripts/resume_after_crash.sh logic: finished stages kept, S5b rerun with the vLLM InternLM2 fix, "
               "S5c finished its remaining pairs, S5d rerun with incremental writes.")
    return out


def status():
    st = repo_path("results/local_eval/stage_status.tsv")
    rows = [l.rstrip("\n").split("\t") for l in st.read_text().splitlines()] if st.exists() else []
    last = {}
    for r in rows:
        if len(r) >= 3 and r[0] != "gpu_temp":
            last[r[0]] = r
    order = ["S1_calibration", "S2_grading", "S3_keyfacts", "S4_claims", "S5a_purge_judge", "S5b_minicheck7b",
             "S5c_flant5", "S5d_nli", "S6_stats", "S8_rag_optional", "S7_pack"]
    fb = json.loads((OUT / "fallbacks.json").read_text()) if (OUT / "fallbacks.json").exists() else []
    ch = json.loads((OUT / "judge_choice.json").read_text()) if (OUT / "judge_choice.json").exists() else {}
    s8 = json.loads((OUT / "S8_rag.json").read_text()) if (OUT / "S8_rag.json").exists() else {}
    probs = json.loads((OUT / "pack_problems.json").read_text()) if (OUT / "pack_problems.json").exists() else []
    info = {}
    for n in ("S3_keyfacts_info", "S4_claims_info", "S2_grade"):
        p = OUT / ".done" / n
        if p.exists():
            try:
                info[n] = json.loads(p.read_text())
            except Exception:
                pass
    L = ["# Local evaluation: STATUS", "", f"_Written {time.strftime('%Y-%m-%d %H:%M')} by `src/local_pack.py status`._", "",
         "| Stage | Result | Wall time (min) | Note |", "|---|---|---|---|"]
    for n in order:
        r = last.get(n)
        if r:
            L.append(f"| {n} | {r[1]} | {r[2]} | {r[3] if len(r) > 3 else ''} |")
        elif (OUT / ".done" / n).exists():
            L.append(f"| {n} | done (earlier run) | – | |")
        else:
            L.append(f"| {n} | not run | – | |")
    L += ["", "## Decisions and fallbacks", ""]
    if ch:
        L.append(f"- Judge: **{ch.get('judge_name')}** prompt {ch.get('prompt')}; gate "
                 f"{'passed' if ch.get('passes_gate') else 'NOT passed → judge grades labelled exploratory'}; candidates "
                 + json.dumps(ch.get("candidates")) + (f"; judge B: {ch['B_skipped']}" if ch.get("B_skipped") else ""))
    for k, v in info.items():
        L.append(f"- {k}: {json.dumps(v)}")
    for f in fb:
        L.append(f"- {f['stage']}: {f['fallback']} ({f['why']})")
    if s8:
        L.append(f"- S8 (RAG baseline): skipped; " + "; ".join(s8.get("reasons", [])) + f". Future work: {s8.get('future_work')}")
    L += gpu_lines()
    sp = repo_path("results/seed_replication/.done")
    done_s = sorted(p.name for p in sp.iterdir()) if sp.exists() else []
    L.append(f"- Seed replication (seed 43, epoch 1; tmux `seeds`): finished steps {', '.join(done_s) or 'none yet'}; "
             "see `results/seed_replication/SUMMARY.md`.")
    if probs:
        L.append("- Paper-pack problems: " + "; ".join(probs))
    L += ["", "## Where to look", "", "- `results/paper_pack/RESULTS.md` (write-up), `results/paper_pack/README.md` (file index)",
          "- `results/local_eval/SUMMARY.md`, `judge_calibration.md`, `keyfact_validation.md`, `metrics_by_system.csv`",
          "- Human review: `results/local_eval/keyfact_spotcheck.xlsx`, `results/paper_pack/human_eval_sheet.xlsx`", ""]
    (OUT / "STATUS.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
    else:
        build()
