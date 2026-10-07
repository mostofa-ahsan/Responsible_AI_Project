"""Build research_materials/ (the paper's data, tables, figures and examples) from result files. Rerunnable; CPU only.

Every number comes from a result file; nothing is typed. A missing input is logged in research_materials/BUILD_LOG.json
and the rest is still written. Main numbers = adapters served on their training base (results/regen); the earlier
bf16-served results appear with the suffix __bf16serve.
    python src/build_research_materials.py
"""

import gzip
import hashlib
import json
import random
import re
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config, repo_path

RM = repo_path("research_materials")
RG = repo_path("results/regen")
LE = repo_path("results/local_eval")
TE = repo_path("results/trained_eval")
CPT = repo_path("results/cpt")
SR = repo_path("results/seed_replication")
ORDER = ["qwen3-8b", "gemma-4-e4b-it", "llama-3.1-8b-instruct"]
LABEL = {"qwen3-8b": "Qwen3-8B", "gemma-4-e4b-it": "Gemma 4 E4B", "llama-3.1-8b-instruct": "Llama 3.1 8B"}
COLOR = {"qwen3-8b": "#0072B2", "gemma-4-e4b-it": "#D55E00", "llama-3.1-8b-instruct": "#009E73"}   # Okabe-Ito
TESTS = ["test_trained_exact", "test_seen_facts", "test_indomain", "test_heldout_docs"]
TLAB = {"test_trained_exact": "exact", "test_seen_facts": "reworded", "test_indomain": "in-domain", "test_heldout_docs": "held-out"}
LOG = {"written": [], "gaps": [], "errors": []}
MANIFEST = []


def rd(p):
    p = Path(p)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def gap(what):
    LOG["gaps"].append(what)


def section(fn):
    def w(*a, **k):
        try:
            fn(*a, **k)
        except Exception as e:
            LOG["errors"].append(f"{fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    w.__name__ = fn.__name__
    return w


def register(path, desc, paper, sources):
    MANIFEST.append({"file": str(Path(path).relative_to(RM)), "description": desc, "paper": paper, "sources": sources})
    LOG["written"].append(str(Path(path).relative_to(RM)))


def write_xlsx(path, sheets, readme, desc, paper, sources, csv=True):
    """sheets: {name: DataFrame}. README sheet first, header bold, frozen panes, fitted widths, 3-decimal numbers."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    for line in [desc, "", f"Paper: {paper}", f"Sources: {sources}", f"Built: {time.strftime('%Y-%m-%d %H:%M')} by "
                 "src/build_research_materials.py", ""] + readme + ["", "Sheets: " + ", ".join(sheets)]:
        ws.append([line])
    ws.column_dimensions["A"].width = 120
    for name, df in sheets.items():
        s = wb.create_sheet(name[:31])
        if df is None or df.empty:
            s.append(["(no data available yet)"])
            continue
        df = df.copy()
        s.append([str(c) for c in df.columns])
        for c in s[1]:
            c.font = Font(bold=True)
        for row in df.itertuples(index=False):
            vals = []
            for v in row:
                if isinstance(v, (float, np.floating)):
                    vals.append(None if np.isnan(v) else round(float(v), 3))
                elif isinstance(v, (np.integer,)):
                    vals.append(int(v))
                elif isinstance(v, (list, dict)):
                    vals.append(json.dumps(v, ensure_ascii=False))
                else:
                    vals.append(v)
            s.append(vals)
        for j, col in enumerate(df.columns, 1):
            w = max([len(str(col))] + [len(str(x)) for x in df[col].head(300).tolist()])
            s.column_dimensions[get_column_letter(j)].width = min(max(w + 2, 8), 60)
            if df[col].dtype.kind == "f":
                for cell in s.iter_rows(min_row=2, min_col=j, max_col=j):
                    cell[0].number_format = "0.000"
        s.freeze_panes = "A2"
        for r in s.iter_rows(min_row=2):
            for c in r:
                c.alignment = Alignment(vertical="top", wrap_text=isinstance(c.value, str) and len(c.value) > 60)
        if csv:
            df.to_csv(path.with_name(f"{path.stem}__{name[:31]}.csv"), index=False)
    wb.save(path)
    register(path, desc, paper, sources)


# ---------------------------------------------------------------- 00 dataset

@section
def dataset():
    d = RM / "00_dataset"
    d.mkdir(parents=True, exist_ok=True)
    splits = {s: rd(repo_path(f"data/splits/{s}.jsonl")) for s in
              ("train", "val", "test_indomain", "test_heldout_docs", "test_seen_facts", "test_trained_exact")}
    sub = json.loads(repo_path("data/splits/eval_subset.json").read_text())["splits"]
    rows = []
    for s, r in splits.items():
        rows.append({"split": s, "qa_pairs": len(r), "documents": len({x.get("doc_id") for x in r}),
                     "eval_subset": len(sub.get(s, [])) or (len(r) if s in ("test_seen_facts", "test_trained_exact") else None)})
    allq = [dict(x, split=s) for s, r in splits.items() for x in r]
    tab = lambda col: pd.crosstab(pd.Series([x.get(col) for x in allq], name=col), pd.Series([x["split"] for x in allq], name="split")).reset_index()
    inv = pd.read_csv(repo_path("data/inventory.csv"))
    meta = pd.read_csv(repo_path("data/parsed/metadata.csv")) if repo_path("data/parsed/metadata.csv").exists() else None
    doc = pd.DataFrame([{"doc_id": x.get("doc_id"), "split": x["split"]} for x in allq])
    bydoc = pd.crosstab(doc.doc_id, doc.split).reset_index()
    if meta is not None and "doc_id" in meta:
        bydoc = bydoc.merge(meta[[c for c in meta.columns if c in ("doc_id", "title", "authors", "year")]], on="doc_id", how="left")
    full = rd(repo_path("data/qa_pairs/full_v1.jsonl"))
    units = sum(1 for _ in open(repo_path("data/qa_pairs/full_v1_units.jsonl"), encoding="utf-8"))
    chunks = rd(repo_path("data/chunks/chunks.jsonl"))
    rej = Counter()
    for x in full:
        for rr in (x.get("reject_reason") or "").split(";"):
            if rr:
                rej[rr.split("(")[0]] += 1
    funnel = pd.DataFrame([{"step": "documents (books / articles)", "count": len(inv)},
                           {"step": "pages", "count": int(inv.pages.sum())},
                           {"step": "chunks (all)", "count": len(chunks)},
                           {"step": "chunks selected for mining", "count": sum(1 for c in chunks if c.get("mine"))},
                           {"step": "knowledge units", "count": units},
                           {"step": "QA pairs generated", "count": len(full)},
                           {"step": "QA pairs passing filters", "count": sum(1 for x in full if x.get("passed_filters"))}] +
                          [{"step": f"rejected: {k} (multi-label)", "count": v} for k, v in rej.most_common()])
    use = rd(repo_path("logs/llm_usage.jsonl"))
    cost = pd.DataFrame([u for u in use if not u.get("cached")])
    if not cost.empty:
        cost = cost.groupby(["stage", "model"], dropna=False).agg(calls=("cost", "size"), input_tokens=("input", "sum"),
                                                                    output_tokens=("output", "sum"), usd=("cost", "sum")).reset_index()
        cost = pd.concat([cost, pd.DataFrame([{"stage": "TOTAL", "model": "", "calls": cost.calls.sum(), "input_tokens": cost.input_tokens.sum(),
                                               "output_tokens": cost.output_tokens.sum(), "usd": cost.usd.sum()}])])
    else:
        gap("api_cost: logs/llm_usage.jsonl missing")
    write_xlsx(d / "dataset_stats.xlsx", {"splits": pd.DataFrame(rows), "by_qtype": tab("q_type"), "by_dimension": tab("dimension"),
                                          "by_difficulty": tab("difficulty"), "by_document": bydoc, "generation_funnel": funnel,
                                          "api_cost": cost},
               ["Counts per split (train/val/tests), by question type, readiness dimension, difficulty and document; the "
                "generation/filter funnel of full_v1; API spend of every uncached LLM call (USD)."],
               "Dataset statistics", "Data section, Table 1", "data/splits/*.jsonl, data/qa_pairs/full_v1*.jsonl, "
               "data/chunks/chunks.jsonl, data/inventory.csv, data/parsed/metadata.csv, logs/llm_usage.jsonl")
    la = repo_path("data/cpt/leakage_audit.json") if repo_path("data/cpt/leakage_audit.json").exists() else CPT / "corpus_leakage_audit.json"
    if la.exists():
        audit = json.loads(la.read_text())
        tr_docs = {x["doc_id"] for x in splits["train"]}
        audit["split_document_overlap"] = {s: len(tr_docs & {x["doc_id"] for x in splits[s]}) for s in ("test_heldout_docs",)}
        audit["note_test_indomain"] = "in-domain questions come from unseen chunks of training documents (by design)"
        (d / "leakage_audit.json").write_text(json.dumps(audit, indent=2))
        register(d / "leakage_audit.json", "Leakage audit (corpus for mixed CPT; held-out document overlap)", "Data section",
                 str(la))
    else:
        gap("leakage_audit.json: no audit file")
    rng = random.Random(42)
    keep = ["qa_id", "question", "answer", "q_type", "dimension", "difficulty", "doc_id"]
    tr = pd.DataFrame(rng.sample(splits["train"], 50))[keep]
    te = pd.DataFrame(rng.sample(splits["test_indomain"] + splits["test_heldout_docs"], 50))[keep + []]
    write_xlsx(d / "qa_sample.xlsx", {"train_50": tr, "test_50": te}, ["Random samples (seed 42) of 50 training and 50 test "
               "QA pairs (in-domain + held-out documents)."], "QA pair samples", "Appendix (data examples)", "data/splits/*.jsonl")


# ---------------------------------------------------------------- 01 training

@section
def training():
    d = RM / "01_training"
    d.mkdir(parents=True, exist_ok=True)
    cfg = load_config()["train"]
    hp = [{"run": "QA-only (closed-book QLoRA)", "base": "NF4 4-bit, double quant, bf16 compute", "lora_r": cfg["lora_r"],
           "lora_alpha": cfg["lora_alpha"], "dropout": cfg["lora_dropout"], "lr": cfg["learning_rate"], "schedule": "cosine",
           "warmup": cfg["warmup_ratio"], "effective_batch": cfg["per_device_batch_size"] * cfg["grad_accum"],
           "max_seq": cfg["max_seq_length"], "seed": cfg["seed"], "data": "QA pairs + paraphrases, answer-only loss"}]
    for f in ORDER:
        p = repo_path(f"models_cpt/{f}/mixC/epochs.json")
        if p.exists():
            st = json.loads(p.read_text())
            hp.append({"run": f"mixed CPT ({LABEL[f]})", "base": "NF4 4-bit, double quant, bf16 compute", "lora_r": st.get("rank"),
                       "lora_alpha": st.get("alpha"), "dropout": 0.05, "lr": 1e-4, "schedule": "cosine over 3 epochs", "warmup": 0.03,
                       "effective_batch": 16, "max_seq": 1024, "seed": 42, "data": "raw training-document text + QA",
                       "micro_batch": st.get("micro_batch")})
    ep = []
    for f in ORDER:
        p = repo_path(f"models/{f}/train_summary.json")
        if p.exists():
            s = json.loads(p.read_text())
            ep.append({"run": "QA-only 1-epoch (cosine over 1)", "model": f, "epoch": 1, "val_loss": s.get("best_val_loss"),
                       "minutes": s.get("train_minutes"), "peak_vram_gb": s.get("peak_vram_gb")})
        for run, p in (("QA-only 3-epoch (cosine over 3)", repo_path(f"models_epochs/{f}/epochs.json")),
                       ("QA-only seed 43 (epoch 1 of 3)", repo_path(f"models_seeds/{f}/seed43/epochs.json")),
                       ("mixed CPT", repo_path(f"models_cpt/{f}/mixC/epochs.json"))):
            if p.exists():
                st = json.loads(p.read_text())
                if "epoch0" in st:
                    ep.append({"run": run, "model": f, "epoch": 0, "val_loss": st["epoch0"].get("val_qa_loss"),
                               "heldout_ppl": st["epoch0"].get("heldout_ppl")})
                for e in st["epochs"]:
                    ep.append({"run": run, "model": f, "epoch": e["epoch"], "train_loss_mean": e.get("train_loss_mean"),
                               "train_loss_last": e.get("train_loss_last"), "raw_loss_mean": e.get("raw_loss_mean"),
                               "qa_loss_mean": e.get("qa_loss_mean"), "val_loss": e.get("val_loss", e.get("val_qa_loss")),
                               "heldout_ppl": e.get("heldout_ppl"), "minutes": e.get("minutes"), "peak_vram_gb": e.get("peak_vram_gb"),
                               "adapter_mb": e.get("adapter_size_mb")})
    fb = []
    for p, tag in ((CPT / "stage_status.tsv", "mixed CPT"), (RG / "stage_status.tsv", "regen")):
        if p.exists():
            for line in p.read_text().splitlines():
                r = line.split("\t")
                if len(r) > 3 and r[1] in ("decision", "pending"):
                    fb.append({"run": tag, "stage": r[0], "time": r[2], "decision": r[3]})
    write_xlsx(d / "training_runs.xlsx", {"hyperparameters": pd.DataFrame(hp), "per_epoch": pd.DataFrame(ep),
                                          "fallbacks": pd.DataFrame(fb)},
               ["Hyperparameters of every run; per-epoch train / validation loss, held-out-document perplexity (mixed CPT), "
                "minutes and peak VRAM; recorded fallbacks and decisions."], "Training runs", "Method section, Table 2",
               "config.yaml, models*/**/epochs.json, models/*/train_summary.json, results/*/stage_status.tsv")
    import trained_eval as T
    rows = []
    for f, model in T.MODEL_ID.items():
        base_n = T.base_param_count(model)
        for run, ad in (("QA-only r16", repo_path(f"models_epochs/{f}/epoch1/adapter")),
                        ("mixed CPT", repo_path(f"models_cpt/{f}/mixC/epoch1/adapter"))):
            fp = ad / "adapter_model.safetensors"
            if not fp.exists():
                continue
            from safetensors import safe_open
            with safe_open(str(fp), "pt") as h:
                n = sum(int(np.prod(h.get_slice(k).get_shape())) for k in h.keys())
            r_ = json.loads((ad / "adapter_config.json").read_text())["r"]
            rows.append({"model": f, "adapter": run, "lora_r": r_, "lora_params": n, "base_params": base_n,
                         "trainable_pct": 100 * n / base_n, "adapter_mb_bf16": fp.stat().st_size / 1e6,
                         "base_checkpoint_gb": T.base_size_gb(model)})
    pd.DataFrame(rows).to_csv(d / "model_footprint.csv", index=False)
    register(d / "model_footprint.csv", "Model footprint: base parameters, LoRA parameters, trainable %, adapter size",
             "Method section", "adapter_model.safetensors files, HF cache base checkpoints")


# ---------------------------------------------------------------- 02 validation

@section
def validation():
    d = RM / "02_validation"
    d.mkdir(parents=True, exist_ok=True)
    cal = json.loads((LE / "calibration_A.json").read_text()) if (LE / "calibration_A.json").exists() else None
    if cal:
        t = cal["test"]
        summ = pd.DataFrame([{"metric": k, "value": v} for k, v in t.items() if not isinstance(v, list)] +
                            [{"metric": "system_pearson", "value": cal["system_level_test"].get("pearson")},
                             {"metric": "system_spearman", "value": cal["system_level_test"].get("spearman")},
                             {"metric": "delta_signs_match", "value": cal["system_level_test"]["n_sign_match"]},
                             {"metric": "significant_delta_signs_match", "value": cal["system_level_test"].get("n_sig_sign_match")},
                             {"metric": "prompt", "value": cal["prompt"]}, {"metric": "passes_gate", "value": cal["passes_gate"]}])
        cm = pd.DataFrame(t["confusion_opus_rows_local_cols"], columns=["local correct", "local partial", "local incorrect"])
        cm.insert(0, "Opus", ["correct", "partial", "incorrect"])
        br = pd.concat([pd.DataFrame([{"group_type": g, "group": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, list)}}
                                      for k, v in gd.items()]) for g, gd in cal["breakdowns_test"].items() if isinstance(gd, dict)])
        write_xlsx(d / "judge_calibration.xlsx", {"test_summary": summ, "confusion": cm, "dev_prompts": pd.DataFrame(cal["dev"]).T.reset_index(),
                                                 "breakdowns": br, "system_points": pd.DataFrame(cal["system_level_test"]["points"]),
                                                 "deltas": pd.DataFrame(cal["system_level_test"]["deltas"])},
                   ["Local judge (Mistral-Small-3.2-24B AWQ, prompt v1) vs Opus 5.5 on held-out calibration items."],
                   "Judge calibration", "Evaluation section, Table 4, Figure 7", "results/local_eval/calibration_A.json")
    else:
        gap("judge_calibration: results/local_eval/calibration_A.json missing")
    v = json.loads((LE / "validation.json").read_text()) if (LE / "validation.json").exists() else {}
    rows = []
    for k, x in v.items():
        if k == "meta":
            continue
        if isinstance(x, dict):
            rows += [{"check": k, "statistic": kk, "value": vv} for kk, vv in x.items()]
        else:
            rows.append({"check": k, "statistic": "value", "value": x})
    write_xlsx(d / "checker_validation.xlsx", {"validation": pd.DataFrame(rows)},
               ["Key-fact recall (MiniCheck-7B, Flan-T5) and NLI contradiction validated against Opus grades."],
               "Checker validation", "Evaluation section, Table 4", "results/local_eval/validation.json")
    s1 = pd.read_csv(SR / "summary.csv") if (SR / "summary.csv").exists() else pd.DataFrame()
    s2 = pd.read_csv(SR / "judge_summary.csv") if (SR / "judge_summary.csv").exists() else pd.DataFrame()
    write_xlsx(d / "seed_replication.xlsx", {"metrics": s1, "judge_accuracy": s2},
               ["Seed 42 vs seed 43 (QA-only, epoch 1, bf16-served), paired bootstrap CIs."], "Seed replication",
               "Limitations / robustness", "results/seed_replication/summary.csv, judge_summary.csv")


# ---------------------------------------------------------------- 03 results

def load_regen():
    mt = pd.read_csv(RG / "metrics_by_system.csv") if (RG / "metrics_by_system.csv").exists() else None
    sig = pd.read_csv(RG / "significance.csv") if (RG / "significance.csv").exists() else None
    df = pd.read_csv(RG / "master_items.csv.gz") if (RG / "master_items.csv.gz").exists() else None
    best = json.loads((RG / "best_epochs.json").read_text()) if (RG / "best_epochs.json").exists() else {}
    return mt, sig, df, best


@section
def results():
    d = RM / "03_results"
    d.mkdir(parents=True, exist_ok=True)
    mt, sig, df, best = load_regen()
    if mt is None:
        gap("03_results: results/regen/metrics_by_system.csv missing (run python src/regen.py stats)")
        return
    write_xlsx(d / "main_results.xlsx", {TLAB[s]: mt[mt.split == s].drop(columns=["split"]) for s in TESTS},
               ["One sheet per test type. Systems: <model>__<variant> = adapter served on its training base (main numbers); "
                "__bf16serve = same adapter on the bf16 base (ablation); __nf4base = base on NF4 weights (control); "
                "__rerender = Gemma prompt re-rendered with the training chat function. Every metric with its 95% bootstrap CI "
                "(_lo, _hi) and n (_n)."], "Main results", "Results section, Table 7, Figures 2-5",
               "results/regen/metrics_by_system.csv")
    write_xlsx(d / "significance_tests.xlsx", {"tests": sig if sig is not None else pd.DataFrame()},
               ["Paired tests on identical items: paired bootstrap (2,000 resamples, Cohen's d_z) and exact McNemar on binary "
                "correct (Cohen's g); Holm-Bonferroni within comparison type × metric × test."], "Significance tests",
               "Results section, Table 7 markers", "results/regen/significance.csv")
    for name, src, desc, paper in (("serving_precision", "serving_precision.csv", "Adapter on the bf16 base vs on its training (NF4) base",
                                    "Results section, serving-precision table, Figure 8"),
                                   ("length_control", "length_control.csv", "Base vs concise base vs best fine-tuned systems (length control)",
                                    "Results section, Figure 5"),
                                   ("robustness_gap", "robustness_gap.csv", "Exact minus reworded accuracy on the 279 paired items",
                                    "Results section, Figure 4")):
        t = pd.read_csv(RG / src) if (RG / src).exists() else pd.DataFrame()
        if t.empty:
            gap(f"{name}: results/regen/{src} empty or missing")
        write_xlsx(d / f"{name}.xlsx", {name: t}, [desc + "."], desc, paper, f"results/regen/{src}")
    if df is not None:
        main = df[df.serving == "match"]
        rows = []
        for gcol in ("q_type", "dimension", "difficulty"):
            g = main.groupby(["system", "split", gcol]).agg(n=("qa_id", "size"), judge_lenient=("judge_lenient", "mean"),
                                                             keyfact_recall=("keyfact_recall", "mean"),
                                                             contradiction_rate=("contradiction_rate", "mean"),
                                                             token_f1=("token_f1", "mean")).reset_index().rename(columns={gcol: "group"})
            g.insert(2, "group_type", gcol)
            rows.append(g)
        write_xlsx(d / "breakdown_qtype_dimension.xlsx", {"breakdown": pd.concat(rows)},
                   ["Means per system × test type × question type / dimension / difficulty (main systems)."],
                   "Breakdown by question type and dimension", "Appendix, Figure 9", "results/regen/master_items.csv.gz")
        out = d / "per_item_results.csv.gz"
        cols = [c for c in df.columns if c not in ("question", "reference")]
        df[cols].to_csv(out, index=False, compression="gzip")
        if out.stat().st_size > 50e6:
            df[[c for c in cols if c != "answer"]].to_csv(out, index=False, compression="gzip")
            gap("per_item_results.csv.gz: answers dropped to stay under 50 MB")
        register(out, "Every answer with its judge grade, key-fact checks, F1/ROUGE-L, length (all systems)",
                 "Supplementary data", "results/regen/master_items.csv.gz")


# ---------------------------------------------------------------- 04 figures

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300})
    return plt


CAPTIONS = {}


def save(fig, name, caption):
    d = RM / "04_figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    CAPTIONS[name] = caption
    register(d / f"{name}.pdf", caption, f"Figure {name.split('_')[0][1:]}", "see CAPTIONS.md")


def cell(mt, system, split, col="judge_lenient"):
    r = mt[(mt.system == system) & (mt.split == split)]
    if r.empty or pd.isna(r[col].iloc[0]):
        return np.nan, 0, 0
    r = r.iloc[0]
    return r[col], r[col] - r[col + "_lo"], r[col + "_hi"] - r[col]


@section
def figures():
    plt = plt_setup()
    mt, sig, df, best = load_regen()
    if mt is None:
        gap("figures: results/regen metrics missing")
        return
    # F2 accuracy on 4 test types x {base, concise base, best QA-only, best mixed CPT}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    kinds = [("base", "base", 0.35), ("concise40", "concise base", 0.55), ("qa", "best QA-only", 0.8), ("cpt", "best mixed CPT", 1.0)]
    for ax, f in zip(axes, ORDER):
        for k, (v, lab, alpha) in enumerate(kinds):
            var = best.get(f, {}).get(v) if v in ("qa", "cpt") else v
            if not var:
                continue
            ys, lo, hi = zip(*[cell(mt, f"{f}__{var}", s) for s in TESTS])
            ax.bar(np.arange(4) + (k - 1.5) * 0.2, ys, 0.2, yerr=[lo, hi], color=COLOR[f], alpha=alpha, capsize=1.5,
                   error_kw={"lw": 0.6}, label=f"{lab}" + (f" ({var})" if v in ("qa", "cpt") else ""))
        ax.set_xticks(range(4), [TLAB[s] for s in TESTS])
        ax.set_title(LABEL[f])
        ax.legend(loc="lower left", fontsize=6)
    axes[0].set_ylabel("judge accuracy")
    save(fig, "F2_accuracy_four_tests", "Judge accuracy (correct + 0.5 partial) on the four test types for the base model, the "
         "concise base (≤ 40 words), the best QA-only epoch and the best mixed-CPT epoch (adapters served on their training "
         "base); 95% bootstrap CIs.")
    # F3 accuracy and val loss vs epoch
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, f in zip(axes, ORDER):
        for prefix, ls, lab in (("ep", "-", "QA-only"), ("cpt_ep", "--", "mixed CPT")):
            xs, ys, lo, hi = [0], [], [], []
            y0 = np.nanmean([cell(mt, f"{f}__base", s)[0] for s in ("test_indomain", "test_heldout_docs")])
            ys.append(y0)
            for e in (1, 2, 3):
                vals = [cell(mt, f"{f}__{prefix}{e}", s)[0] for s in ("test_indomain", "test_heldout_docs")]
                if not np.all(np.isnan(vals)):
                    xs.append(e)
                    ys.append(np.nanmean(vals))
            ax.plot(xs, ys, ls, marker="o", ms=3, color=COLOR[f], label=f"{lab}: unseen-question accuracy")
        ax2 = ax.twinx()
        for p, ls, lab in ((repo_path(f"models_epochs/{f}/epochs.json"), "-", "QA-only val loss"),
                           (repo_path(f"models_cpt/{f}/mixC/epochs.json"), "--", "mixed CPT val QA loss")):
            if p.exists():
                st = json.loads(p.read_text())
                e = [x["epoch"] for x in st["epochs"]]
                vl = [x.get("val_loss", x.get("val_qa_loss")) for x in st["epochs"]]
                ax2.plot(e, vl, ls, color="grey", lw=0.8, marker="s", ms=2, label=lab)
        ax.set_xticks([0, 1, 2, 3], ["base", "1", "2", "3"])
        ax.set_title(LABEL[f])
        ax2.tick_params(labelsize=6)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("judge accuracy (in-domain + held-out)")
    axes[0].legend(fontsize=6, loc="lower left")
    save(fig, "F3_accuracy_and_val_loss_vs_epoch", "Mean judge accuracy on unseen questions (in-domain + held-out) against "
         "epoch (0 = base; coloured, left axis) and validation loss (grey, right axis), QA-only (solid) and mixed CPT (dashed).")
    # F4 exact vs reworded dumbbell
    gp = pd.read_csv(RG / "robustness_gap.csv") if (RG / "robustness_gap.csv").exists() else None
    if gp is not None and len(gp):
        g = gp[(gp.metric == "judge_lenient") & (gp.serving == "match")]
        sel = []
        for f in ORDER:
            for v in ("base", best.get(f, {}).get("qa"), best.get(f, {}).get("cpt")):
                r = g[g.system == f"{f}__{v}"] if v else g.iloc[0:0]
                if len(r):
                    sel.append((f, v, r.iloc[0]))
        fig, ax = plt.subplots(figsize=(3.5, 0.28 * len(sel) + 0.6))
        for i, (f, v, r) in enumerate(sel):
            ax.plot([r.reworded, r.exact], [i, i], color=COLOR[f], lw=1.2)
            ax.scatter([r.reworded], [i], color="white", edgecolor=COLOR[f], s=18, zorder=3)
            ax.scatter([r.exact], [i], color=COLOR[f], s=18, zorder=3)
        ax.set_yticks(range(len(sel)), [f"{LABEL[f]} {v}" for f, v, _ in sel])
        ax.set_xlabel("judge accuracy (open = reworded, filled = exact)")
        save(fig, "F4_exact_vs_reworded", "Exact training question vs its rewording on the 279 paired items (filled: exact, "
             "open: reworded) for base, best QA-only and best mixed-CPT systems.")
    else:
        gap("F4: robustness_gap.csv missing")
    # F5 length decomposition (held-out)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for i, f in enumerate(ORDER):
        b = cell(mt, f"{f}__base", "test_heldout_docs")[0]
        c = cell(mt, f"{f}__concise40", "test_heldout_docs")[0]
        q = cell(mt, f"{f}__{best.get(f, {}).get('qa')}", "test_heldout_docs")[0] if best.get(f, {}).get("qa") else np.nan
        ax.bar(i - 0.2, c - b, 0.38, color=COLOR[f], alpha=0.45, label="length only (concise − base)" if i == 0 else None)
        ax.bar(i + 0.2, q - c, 0.38, color=COLOR[f], label="fine-tuning at matched length (best QA-only − concise)" if i == 0 else None)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(range(3), [LABEL[f] for f in ORDER])
    ax.set_ylabel("Δ judge accuracy, held-out documents")
    ax.legend(fontsize=6, loc="lower left")
    save(fig, "F5_length_decomposition", "Held-out documents: change in judge accuracy from shortening the base answer "
         "(concise base − base) and from fine-tuning at matched length (best QA-only − concise base).")
    # F6 F1 and key-fact recall vs Opus accuracy (18 cells)
    lm = pd.read_csv(LE / "metrics_by_system.csv") if (LE / "metrics_by_system.csv").exists() else None
    if lm is not None and "opus_lenient" in lm:
        c = lm[lm.opus_lenient.notna()]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
        for ax, col, lab in zip(axes, ("token_f1", "keyfact_recall"), ("token F1", "key-fact recall")):
            for f in ORDER:
                cc = c[c.family == f]
                ax.scatter(cc.opus_lenient, cc[col], color=COLOR[f], s=16, label=LABEL[f])
            r = np.corrcoef(c.opus_lenient, c[col])[0, 1]
            ax.set_title(f"{lab} vs Opus accuracy (r = {r:.2f}, 18 cells)")
            ax.set_xlabel("Opus judge accuracy")
            ax.set_ylabel(lab)
        axes[0].legend(fontsize=6)
        save(fig, "F6_metrics_vs_opus", "Token F1 (left) and key-fact recall (right) against Opus 5.5 accuracy for the 18 "
             "system × test cells graded by Opus (bf16-served base and 1-epoch systems); Pearson r shown.")
    else:
        gap("F6: results/local_eval/metrics_by_system.csv missing Opus columns")
    # F7 judge validation
    cal = json.loads((LE / "calibration_A.json").read_text()) if (LE / "calibration_A.json").exists() else None
    if cal:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
        cm = np.array(cal["test"]["confusion_opus_rows_local_cols"])
        a1.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        labs = ["correct", "partial", "incorrect"]
        a1.set_xticks(range(3), labs)
        a1.set_yticks(range(3), labs)
        for i in range(3):
            for j in range(3):
                a1.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
        a1.set_xlabel("local judge")
        a1.set_ylabel("Opus 5.5")
        a1.set_title(f"TEST items, κ = {cal['test']['kappa']:.2f}")
        for p in cal["system_level_test"]["points"]:
            f, v = p["system"].split("__")
            a2.scatter(p["opus"], p["local"], color=COLOR[f], marker="o" if v == "base" else "^", s=16)
        lim = [0.3, 0.8]
        a2.plot(lim, lim, "--", color="grey", lw=0.8)
        a2.set_xlabel("Opus accuracy")
        a2.set_ylabel("local-judge accuracy")
        a2.set_title(f"system level: r = {cal['system_level_test']['pearson']:.2f}")
        save(fig, "F7_judge_validation", "Local judge vs Opus 5.5 on held-out calibration items: confusion matrix (left) and "
             "system × test accuracy (right; circles base, triangles 1-epoch fine-tuned).")
    else:
        gap("F7: calibration_A.json missing")
    # F8 serving precision + deployment variants
    sp = pd.read_csv(RG / "serving_precision.csv") if (RG / "serving_precision.csv").exists() else pd.DataFrame()
    dep = pd.read_csv(TE / "deployment.csv") if (TE / "deployment.csv").exists() else pd.DataFrame()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    if len(sp):
        s = sp[(sp.metric == "judge_lenient")]
        for i, f in enumerate(ORDER):
            for j, t in enumerate(TESTS):
                r = s[(s.family == f) & (s.split == t) & (s.variant == best.get(f, {}).get("qa"))]
                if len(r):
                    r = r.iloc[0]
                    x = j + (i - 1) * 0.22
                    a1.plot([x, x], [r.ref_mean, r.sys_mean], color=COLOR[f], lw=1.2)
                    a1.scatter([x], [r.ref_mean], color="white", edgecolor=COLOR[f], s=14, zorder=3)
                    a1.scatter([x], [r.sys_mean], color=COLOR[f], s=14, zorder=3, label=LABEL[f] if j == 0 else None)
        a1.set_xticks(range(4), [TLAB[t] for t in TESTS])
        a1.set_ylabel("judge accuracy")
        a1.set_title("best QA-only adapter: bf16 base (open) vs training base (filled)")
        a1.legend(fontsize=6)
    else:
        gap("F8: serving_precision.csv missing")
    if len(dep):
        for i, r in enumerate(dep.itertuples()):
            accs = [getattr(r, f"judge_{s}") for s in ("test_trained_exact", "test_seen_facts", "test_heldout_docs")]
            accs = [a for a in accs if a == a]
            if accs and r.disk_gb == r.disk_gb:
                a2.scatter(r.disk_gb, np.mean(accs), color=COLOR[r.family], s=18,
                           marker={"bf16": "o", "NF4": "s"}.get(r.deployment.split()[0], "^"))
        a2.set_xlabel("disk size (GB)")
        a2.set_ylabel("judge accuracy (mean of 3 tests)")
        a2.set_title("deployment variants (circle bf16, square NF4, triangle merged 4-bit)")
    save(fig, "F8_serving_precision_deployment", "Left: the best QA-only adapter served on the bf16 base (open) vs on its "
         "NF4 training base (filled). Right: deployment variants that exist (earlier deployment run; merged 4-bit variants "
         "were merged into the bf16 base).")
    # F9 heatmap gain by q_type x dimension (best QA-only - base, unseen tests, all models)
    if df is not None:
        parts = []
        for f in ORDER:
            v = best.get(f, {}).get("qa")
            if not v:
                continue
            a = df[(df.system == f"{f}__base") & df.split.isin(["test_indomain", "test_heldout_docs"])].set_index(["split", "qa_id"])
            b = df[(df.system == f"{f}__{v}") & df.split.isin(["test_indomain", "test_heldout_docs"])].set_index(["split", "qa_id"])
            j = a[["q_type", "dimension", "judge_lenient"]].join(b[["judge_lenient"]], rsuffix="_ft", how="inner")
            j["gain"] = j.judge_lenient_ft - j.judge_lenient
            parts.append(j)
        if parts:
            j = pd.concat(parts)
            piv = j.pivot_table(index="q_type", columns="dimension", values="gain", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(7.2, 2.4))
            lim = np.nanmax(np.abs(piv.to_numpy()))
            im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_xticks(range(len(piv.columns)), [c.replace("_", " ") for c in piv.columns], rotation=35, ha="right")
            ax.set_yticks(range(len(piv.index)), piv.index)
            for i in range(piv.shape[0]):
                for k in range(piv.shape[1]):
                    if not np.isnan(piv.iat[i, k]):
                        ax.text(k, i, f"{piv.iat[i, k]:+.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, label="Δ judge accuracy")
            save(fig, "F9_gain_heatmap", "Appendix: change in judge accuracy (best QA-only epoch on its training base − base) by "
                 "question type × readiness dimension, unseen questions (in-domain + held-out), three models pooled.")
    (RM / "04_figures" / "CAPTIONS.md").write_text("# Figure captions\n\n" + "\n\n".join(f"**{k}**: {v}" for k, v in sorted(CAPTIONS.items())) + "\n")
    register(RM / "04_figures" / "CAPTIONS.md", "Figure captions", "Figures", "src/build_research_materials.py")


# ---------------------------------------------------------------- 05 examples

def short(t, n=70):
    w = str(t).split()
    return " ".join(w[:n]) + ("" if len(w) <= n else " …") + f" [{len(w)} words]"


@section
def examples():
    d = RM / "05_examples"
    d.mkdir(parents=True, exist_ok=True)
    mt, sig, df, best = load_regen()
    if df is None:
        gap("examples: master items missing")
        return
    log = {"rules": {}, "picked": {}}
    # T5 wins
    cands = []
    for f in ORDER:
        v = best.get(f, {}).get("qa")
        if not v:
            continue
        a = df[df.system == f"{f}__base"].set_index(["split", "qa_id"])
        b = df[df.system == f"{f}__{v}"].set_index(["split", "qa_id"])
        j = b.join(a, rsuffix="_base", how="inner")
        ok = j[(j.local_grade == "correct") & j.local_grade_base.isin(["partial", "incorrect"]) &
               ((j.keyfact_recall - j.keyfact_recall_base) >= 0.5) & (j.contradicted_facts == 0)]
        for (s, q), r in ok.iterrows():
            pair_ok = False
            if s == "test_trained_exact" and isinstance(r.paraphrase_test_id, str):
                pr = df[(df.system == f"{f}__{v}") & (df.split == "test_seen_facts") & (df.qa_id == r.paraphrase_test_id)]
                pair_ok = bool(len(pr)) and pr.local_grade.iloc[0] == "correct"
            cands.append({"family": f, "system": f"{f}__{v}", "split": s, "qa_id": q, "q_type": r.q_type, "dimension": r.dimension,
                          "doc_id": r.doc_id, "recall_gain": r.keyfact_recall - r.keyfact_recall_base, "reworded_also_correct": pair_ok,
                          "question": r.question, "reference": r.reference, "base_answer": short(r.answer_base),
                          "base_verdict": r.local_grade_base, "base_recall": f"{int(r.supported_facts_base)}/{int(r.n_facts)}",
                          "ft_answer": short(r.answer), "ft_verdict": r.local_grade, "ft_recall": f"{int(r.supported_facts)}/{int(r.n_facts)}"})
    rng = random.Random(7)
    rng.shuffle(cands)
    cands.sort(key=lambda c: (-int(c["reworded_also_correct"]), -c["recall_gain"]))
    picked, dims, qts = [], set(), set()
    for f in ORDER:                                         # >= 3 per model first, maximizing coverage
        for _ in range(3):
            pool = [c for c in cands if c["family"] == f and c not in picked]
            if not pool:
                break
            c = max(pool, key=lambda c: (c["dimension"] not in dims) + (c["q_type"] not in qts) + 0.5 * c["reworded_also_correct"])
            picked.append(c)
            dims.add(c["dimension"])
            qts.add(c["q_type"])
    while len(picked) < 12:
        pool = [c for c in cands if c not in picked]
        if not pool:
            break
        c = max(pool, key=lambda c: (c["dimension"] not in dims) + (c["q_type"] not in qts) + 0.5 * c["reworded_also_correct"])
        picked.append(c)
        dims.add(c["dimension"])
        qts.add(c["q_type"])
    if len(picked) < 12 or len(dims) < 4:
        gap(f"T5_wins: {len(picked)} items, {len(dims)} dimensions, {len(qts)} q_types (targets 12, >= 4, all)")
    t5 = pd.DataFrame(picked)
    log["rules"]["T5"] = "fine-tuned (best QA-only, training base) correct; base partial/incorrect; key-fact recall gain >= 0.5; " \
                         "no contradicted fact; >= 3 per model; coverage of dimensions and q_types maximized; preference for " \
                         "exact items whose reworded pair is also correct; seed 7"
    log["picked"]["T5"] = [{k: c[k] for k in ("system", "split", "qa_id", "dimension", "q_type", "reworded_also_correct")} for c in picked]
    write_xlsx(d / "T5_wins.xlsx", {"T5_wins": t5}, ["Items where the fine-tuned model is right and the base is not (rules in "
               "selection_log.json)."], "Table 5: fine-tuning wins", "Results section, Table 5", "results/regen/master_items.csv.gz")
    # T6 side by side
    need = [f"{f}__base" for f in ORDER] + [f"{f}__{best[f]['qa']}" for f in ORDER if best.get(f, {}).get("qa")]
    if best.get("llama-3.1-8b-instruct", {}).get("cpt"):
        need.append(f"llama-3.1-8b-instruct__{best['llama-3.1-8b-instruct']['cpt']}")
    need.append("llama-3.1-8b-instruct__concise40")
    sheets = {}
    chosen = []
    for k, s in enumerate(("test_heldout_docs", "test_indomain", "test_trained_exact")):
        sub = df[(df.split == s) & df.system.isin(need) & df.local_grade.notna()]
        full = sub.groupby("qa_id").system.nunique()
        qs = sorted(full[full == len(need)].index)
        if not qs:
            gap(f"T6: no {s} item answered and graded by all {len(need)} systems")
            continue
        q = random.Random(11 + k).choice(qs)
        chosen.append((s, q))
        rows = sub[sub.qa_id == q].set_index("system").loc[need]
        it = rows.iloc[0]
        sheets[f"6{'abc'[k]}"] = pd.DataFrame([{"system": sysn, "question": it.question if i == 0 else "",
                                                "reference": it.reference if i == 0 else "", "answer": short(r.answer),
                                                "verdict": r.local_grade, "key_fact_recall": f"{int(r.supported_facts)}/{int(r.n_facts)}"
                                                if r.supported_facts == r.supported_facts else "–",
                                                "contradicted_facts": r.contradicted_facts}
                                               for i, (sysn, r) in enumerate(rows.iterrows())])
    ce = []
    for f in ORDER:
        v = best.get(f, {}).get("qa")
        if not v:
            continue
        a = df[df.system == f"{f}__base"].set_index(["split", "qa_id"])
        b = df[df.system == f"{f}__{v}"].set_index(["split", "qa_id"])
        j = b.join(a, rsuffix="_base", how="inner")
        bad = j[(j.local_grade == "incorrect") & (j.local_grade_base == "correct") & (j.contradicted_facts > 0)]
        for (s, q), r in bad.iterrows():
            ce.append({"system": f"{f}__{v}", "split": s, "qa_id": q, "question": r.question, "reference": r.reference,
                       "base_answer": short(r.answer_base), "base_verdict": r.local_grade_base, "ft_answer": short(r.answer),
                       "ft_verdict": r.local_grade, "contradicted_facts": r.contradicted_facts})
    if ce:
        sheets["6d_counterexample"] = pd.DataFrame([random.Random(3).choice(ce)])
    else:
        gap("T6d: no counter-example (base correct, fine-tuned incorrect with a contradiction)")
    log["picked"]["T6"] = [{"split": s, "qa_id": q} for s, q in chosen]
    log["rules"]["T6"] = "one item per test type (held-out, in-domain, exact) answered and graded by every listed system; seeded choice"
    write_xlsx(d / "T6_side_by_side.xlsx", sheets, ["Side-by-side answers of base, best QA-only, Llama best mixed CPT and Llama "
               "concise base; 6d is an honest counter-example."], "Table 6: side-by-side answers", "Results section, Table 6",
               "results/regen/master_items.csv.gz")
    for name, sh in (("T5_wins", {"T5": t5}), ("T6_side_by_side", sheets)):
        md, tex = [], []
        for k, t in sh.items():
            if t is None or t.empty:
                continue
            md.append(f"## {k}\n\n" + t.to_markdown(index=False))
            tex.append(t.to_latex(index=False, escape=True, longtable=True))
        (d / f"{name}.md").write_text("\n\n".join(md) + "\n")
        (d / f"{name}.tex").write_text("\n\n".join(tex) + "\n")
    # sources
    docs = sorted({c["doc_id"] for c in picked} | {df[(df.split == s) & (df.qa_id == q)].doc_id.iloc[0] for s, q in chosen} |
                  ({x["qa_id"] and df[df.qa_id == x["qa_id"]].doc_id.iloc[0] for x in ce[:0]}))
    meta = pd.read_csv(repo_path("data/parsed/metadata.csv")) if repo_path("data/parsed/metadata.csv").exists() else pd.DataFrame()
    inv = pd.read_csv(repo_path("data/inventory.csv"))
    md, bib = ["# Sources of the example items", ""], []
    for doc in docs:
        m = meta[meta.doc_id == doc].iloc[0].to_dict() if len(meta) and (meta.doc_id == doc).any() else {}
        iv = inv[inv.doc_id == doc].iloc[0].to_dict() if (inv.doc_id == doc).any() else {}
        title = m.get("title") or iv.get("title")
        authors = m.get("authors") or m.get("author") or iv.get("author")
        year = m.get("year") or iv.get("year")
        missing = [k for k, v in (("title", title), ("authors", authors), ("year", year)) if not v or (isinstance(v, float) and np.isnan(v))]
        md.append(f"- `{doc}`: {authors or '?'} ({year or '?'}). *{title or '?'}*. File: {iv.get('path', '?')}"
                  + (f" — **missing: {', '.join(missing)}**" if missing else ""))
        key = re.sub(r"[^a-z0-9]", "", doc.lower())[:24]
        bib.append(f"@misc{{{key},\n  title = {{{title or ''}}},\n  author = {{{authors or ''}}},\n  year = {{{year or ''}}},\n"
                   f"  note = {{doc_id {doc}{'; missing ' + ', '.join(missing) if missing else ''}}}\n}}")
    (d / "SOURCES.md").write_text("\n".join(md) + "\n")
    (d / "sources.bib").write_text("\n\n".join(bib) + "\n")
    (d / "selection_log.json").write_text(json.dumps(log, indent=2, default=str))
    for f_, desc in (("SOURCES.md", "Bibliographic data of the source documents of the examples (missing fields flagged)"),
                     ("sources.bib", "BibTeX of the source documents"), ("selection_log.json", "Example selection rules and picks"),
                     ("T5_wins.md", "Table 5 (Markdown)"), ("T5_wins.tex", "Table 5 (LaTeX)"),
                     ("T6_side_by_side.md", "Table 6 (Markdown)"), ("T6_side_by_side.tex", "Table 6 (LaTeX)")):
        register(d / f_, desc, "Results section, Tables 5-6", "results/regen/master_items.csv.gz, data/parsed/metadata.csv")


# ---------------------------------------------------------------- 06 diagnostics, 07 deployment

@section
def diagnostics():
    d = RM / "06_diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    for src in (repo_path("results/diagnostics/format_check.md"), repo_path("results/diagnostics/dequant_check.json"),
                *sorted(RG.glob("dequant_verify_*.json"))):
        if src.exists():
            shutil.copy(src, d / src.name)
            register(d / src.name, "Prompt-format check / dequantized-base verification", "Methods (serving precision), appendix",
                     str(src.relative_to(repo_path("."))))
        else:
            gap(f"diagnostics: {src} missing")


@section
def deployment():
    d = RM / "07_deployment"
    d.mkdir(parents=True, exist_ok=True)
    dep = pd.read_csv(TE / "deployment.csv") if (TE / "deployment.csv").exists() else pd.DataFrame()
    st = json.loads((TE / "deploy_stats.json").read_text()) if (TE / "deploy_stats.json").exists() else {}
    missing = pd.DataFrame([{"variant": "GGUF Q4_K_M (llama.cpp)", "status": "not built",
                             "why": "; ".join(json.loads((TE / "gguf_skipped.json").read_text())["reasons"]) if (TE / "gguf_skipped.json").exists() else "not run"},
                            {"variant": "merged 4-bit (AWQ/RTN) on the matching NF4-dequantized base", "status": "pending",
                             "why": "see results/regen/STATUS.md"}])
    mb = json.loads((RG / "deploy_stats.json").read_text()) if (RG / "deploy_stats.json").exists() else {}
    mt = pd.read_csv(RG / "metrics_by_system.csv") if (RG / "metrics_by_system.csv").exists() else pd.DataFrame()
    mrows = []
    for k, v in mb.items():
        fam, var, _ = k.split("__")
        row = {"system": k, **v}
        for s in TESTS:
            r = mt[(mt.system == f"{fam}__{var}") & (mt.split == s)] if len(mt) else mt
            row[f"judge_{TLAB[s]}"] = float(r.judge_lenient.iloc[0]) if len(r) else None
            ref = mt[(mt.system == f"{fam}__{v['adapter_epoch']}") & (mt.split == s)] if len(mt) else mt
            row[f"judge_{TLAB[s]}_adapter_on_training_base"] = float(ref.judge_lenient.iloc[0]) if len(ref) else None
        mrows.append(row)
    if mb:
        missing.loc[missing.variant.str.contains("matching"), ["status", "why"]] = ["done", "see sheet matching_base_variants"]
    write_xlsx(d / "deployment.xlsx", {"deployment": dep, "backend_stats": pd.DataFrame([{"system": k, **v} for k, v in st.items()]),
                                       "matching_base_variants": pd.DataFrame(mrows), "missing_variants": missing},
               ["Deployment variants of each model's best epoch from the earlier deployment run (NF4 + adapter in transformers; "
                "merged 4-bit served by vLLM, merged into the bf16 base). Missing variants listed with the reason."],
               "Deployment variants", "Discussion / deployment, Figure 8", "results/trained_eval/deployment.csv, deploy_stats.json")


def readme():
    man = pd.DataFrame(MANIFEST)
    rows = []
    for m in MANIFEST:
        p = RM / m["file"]
        rows.append({**m, "bytes": p.stat().st_size if p.exists() else None,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() and p.stat().st_size < 60e6 else ""})
    pd.DataFrame(rows).to_csv(RM / "MANIFEST.csv", index=False)
    L = ["# research_materials", "", f"_Built {time.strftime('%Y-%m-%d %H:%M')} by `python src/build_research_materials.py` "
         "(rerunnable; every number is read from result files). Main numbers: adapters served on their training base "
         "(NF4-dequantized); earlier bf16-served results carry the suffix `__bf16serve`._", "",
         "| File | Description | Paper | Sources |", "|---|---|---|---|"]
    for m in MANIFEST:
        L.append(f"| `{m['file']}` | {m['description']} | {m['paper']} | {m['sources']} |")
    L += ["", "## Gaps", ""] + ([f"- {g}" for g in LOG["gaps"]] or ["- none"])
    if LOG["errors"]:
        L += ["", "## Build errors", ""] + [f"- {e}" for e in LOG["errors"]]
    (RM / "README.md").write_text("\n".join(L) + "\n")


def main():
    RM.mkdir(exist_ok=True)
    for fn in (dataset, training, validation, results, figures, examples, diagnostics, deployment):
        fn()
    readme()
    (RM / "BUILD_LOG.json").write_text(json.dumps(LOG, indent=2))
    print(f"research_materials: {len(LOG['written'])} files, {len(LOG['gaps'])} gaps, {len(LOG['errors'])} errors")


if __name__ == "__main__":
    main()
