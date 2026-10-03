"""Results for a multi-model run: per-run summaries, comparison tables, plot, environment.

Reads results/<run>/<model>/<arm>/<split>_predictions.jsonl and _grades.jsonl. Token F1 and
ROUGE-L use every test question; judge accuracy (correct = 1, partial = 0.5) uses the graded
subset, which is the same questions for every model and arm. CIs are percentile bootstrap 95%
(2,000 resamples); improvements use a paired bootstrap over the same questions.
"""

import csv
import json
import platform
import random
import subprocess
from collections import defaultdict

from eval_closedbook import rouge_l, token_f1
from qa_common import read_jsonl
from utils import repo_path

SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
N_BOOT = 2000


def boot(values, seed=0):
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return {"mean": round(sum(values) / n, 4), "lo": round(means[int(0.025 * N_BOOT)], 4),
            "hi": round(means[int(0.975 * N_BOOT) - 1], 4), "n": n}


def paired_diff(a, b, seed=0):
    """b - a over paired values (dicts keyed by qa_id, same keys used)."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    d = [b[k] - a[k] for k in keys]
    return boot(d, seed)


def fmt(x, digits=3):
    return "-" if x is None or x.get("mean") is None else f"{x['mean']:.{digits}f} [{x['lo']:.{digits}f}, {x['hi']:.{digits}f}]"


def graded_ids(rd, cfg, split):
    """Questions in the (possibly shrunk) fixed subset: identical for every model and arm."""
    sub = json.loads((repo_path(cfg["paths"]["splits"]) / "eval_subset.json").read_text())["splits"][split]
    plan = rd / "grading_plan.json"
    k = json.loads(plan.read_text())["subset_k"][split] if plan.exists() else len(sub)
    return set(sub[:k])


def per_run(rd, model_short, arm, split, refs, keep_ids):
    d = rd / model_short / arm
    pp = d / f"{split}_predictions.jsonl"
    if not pp.exists():
        return None
    preds = {x["qa_id"]: x["prediction"] for x in read_jsonl(pp)}
    grade_rows = [x for x in read_jsonl(d / f"{split}_grades.jsonl") if keep_ids is None or x["qa_id"] in keep_ids] \
        if (d / f"{split}_grades.jsonl").exists() else []
    grades = {x["qa_id"]: x["verdict"] for x in grade_rows}
    unsupported = {x["qa_id"]: x["unsupported_claims"] for x in grade_rows if "unsupported_claims" in x}
    f1 = {q: token_f1(p, refs[q]["answer"]) for q, p in preds.items()}
    rl = {q: rouge_l(p, refs[q]["answer"]) for q, p in preds.items()}
    js = {q: SCORE[v] for q, v in grades.items()}

    def block(qs):
        qs = list(qs)
        return {"judge_accuracy": boot([js[q] for q in qs if q in js]),
                "token_f1": boot([f1[q] for q in qs]), "rouge_l": boot([rl[q] for q in qs])}
    by_type, by_dim = defaultdict(list), defaultdict(list)
    for q in preds:
        by_type[refs[q]["q_type"]].append(q)
        by_dim[refs[q]["dimension"]].append(q)
    gen = d / f"{split}_generation.json"
    summary = {"model": model_short, "arm": arm, "split": split, "n_questions": len(preds), "n_graded": len(js),
               "generation": json.loads(gen.read_text()) if gen.exists() else None,
               "overall": block(preds), "by_q_type": {k: block(v) for k, v in sorted(by_type.items())},
               "by_dimension": {k: block(v) for k, v in sorted(by_dim.items())},
               "verdicts": {v: sum(1 for x in grades.values() if x == v) for v in SCORE}}
    if unsupported:
        summary["unsupported_claims"] = {k: sum(1 for v in unsupported.values() if v == k) for k in ("0", "1", "2+")}
    (d / f"{split}_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "js": js, "f1": f1, "rl": rl}


def environment(rd, cfg, models):
    lines = []

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"(failed: {e})"
    lines.append(f"python: {platform.python_version()}  platform: {platform.platform()}")
    lines.append("GPU: " + sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"))
    lines.append("git commit: " + sh("git rev-parse HEAD") + (" (dirty)" if sh("git status --porcelain") else ""))
    keys = ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets", "anthropic",
            "sentence-transformers", "numpy")
    lines.append("\nmain venv (.venv):")
    for line in sh(".venv/bin/pip list --format=freeze 2>/dev/null").splitlines():
        if line.split("==")[0].lower() in keys:
            lines.append(f"  {line}")
    lines.append("vLLM venv (.venv-vllm):")
    for line in sh(".venv-vllm/bin/pip list --format=freeze 2>/dev/null").splitlines():
        if line.split("==")[0].lower() in ("vllm", "torch", "transformers"):
            lines.append(f"  {line}")
    lines.append(f"\nseeds: train.seed={cfg['train']['seed']}, split.seed={cfg['split']['seed']}, eval subset seed=42, "
                 f"bootstrap seed=0 ({N_BOOT} resamples)")
    lines.append(f"decoding: greedy, max_new_tokens={cfg['eval']['max_new_tokens']}, thinking disabled (Qwen3)")
    lines.append(f"judge: {cfg['models']['judge']['model']} (effort {cfg['models']['judge'].get('effort')}), "
                 "Message Batches API")
    lines.append("\nmodel revisions (HF cache snapshot):")
    try:
        from huggingface_hub import scan_cache_dir
        info = {r.repo_id: r for r in scan_cache_dir().repos}
        for m in models:
            r = info.get(m)
            lines.append(f"  {m}: " + (", ".join(rv.commit_hash for rv in r.revisions) if r else "not cached"))
    except Exception as e:  # noqa: BLE001
        lines.append(f"  (scan failed: {e})")
    lines.append("\ntraining recipe: " + json.dumps({k: cfg["train"][k] for k in (
        "lora_r", "lora_alpha", "lora_dropout", "learning_rate", "per_device_batch_size", "grad_accum",
        "max_seq_length", "warmup_ratio", "seed", "use_paraphrases")}))
    (rd / "environment.txt").write_text("\n".join(lines) + "\n")


def plot(rd, rows, splits=("test_indomain", "test_heldout_docs")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in rows}, key=lambda m: [x["model"] for x in rows].index(m))
    fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 4.2), sharey=True)
    colors = {"base": "#9aa5b1", "finetuned": "#2f6db3"}
    for ax, split in zip(axes, splits):
        for j, arm in enumerate(("base", "finetuned")):
            xs, ys, err = [], [], [[], []]
            for i, m in enumerate(models):
                r = next((x for x in rows if x["model"] == m and x["split"] == split and x["arm"] == arm), None)
                if not r or r["judge_mean"] is None:
                    continue
                xs.append(i + (j - 0.5) * 0.38)
                ys.append(r["judge_mean"])
                err[0].append(r["judge_mean"] - r["judge_lo"])
                err[1].append(r["judge_hi"] - r["judge_mean"])
            ax.bar(xs, ys, width=0.36, yerr=err, capsize=3, color=colors[arm], label=arm)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=0, fontsize=9)
        ax.set_title(split)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Opus-judged accuracy (95% CI)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Closed-book accuracy: base vs QLoRA fine-tuned")
    fig.tight_layout()
    fig.savefig(rd / "judge_accuracy.png", dpi=150)
    plt.close(fig)


def write_all(cfg, run, models, arms, splits, log):
    rd = repo_path("results") / run
    refs = {}
    for split in splits:
        for r in read_jsonl(repo_path(cfg["paths"]["splits"]) / f"{split}.jsonl"):
            refs[r["qa_id"]] = r
    seen_path = repo_path(cfg["paths"]["splits"]) / "test_seen_facts.jsonl"
    has_seen = seen_path.exists() and any((rd / m.split("/")[-1].lower() / a / "test_seen_facts_predictions.jsonl").exists()
                                          for m in models for a in arms)
    if has_seen:
        for r in read_jsonl(seen_path):
            refs[r["qa_id"]] = r
    all_splits = list(splits) + (["test_seen_facts"] if has_seen else [])
    data, rows = {}, []
    for m in models:
        ms = m.split("/")[-1].lower()
        for arm in arms:
            for split in all_splits:
                keep = None if split == "test_seen_facts" else graded_ids(rd, cfg, split)
                r = per_run(rd, ms, arm, split, refs, keep)
                if r:
                    data[(ms, arm, split)] = r
                    o = r["summary"]["overall"]
                    rows.append({"model": ms, "arm": arm, "split": split, "n_questions": r["summary"]["n_questions"],
                                 "n_graded": r["summary"]["n_graded"],
                                 "judge_mean": o["judge_accuracy"]["mean"], "judge_lo": o["judge_accuracy"]["lo"],
                                 "judge_hi": o["judge_accuracy"]["hi"],
                                 "f1_mean": o["token_f1"]["mean"], "f1_lo": o["token_f1"]["lo"], "f1_hi": o["token_f1"]["hi"],
                                 "rougeL_mean": o["rouge_l"]["mean"], "rougeL_lo": o["rouge_l"]["lo"],
                                 "rougeL_hi": o["rouge_l"]["hi"]})
    train = {}
    for m in models:
        ms = m.split("/")[-1].lower()
        p = repo_path("models") / ms / "train_summary.json"
        if p.exists():
            train[ms] = json.loads(p.read_text())
    # improvements (paired)
    imp = {}
    for (ms, arm, split), r in data.items():
        if arm != "finetuned" or (ms, "base", split) not in data:
            continue
        b = data[(ms, "base", split)]
        imp[(ms, split)] = {"judge_accuracy": paired_diff(b["js"], r["js"]), "token_f1": paired_diff(b["f1"], r["f1"]),
                            "rouge_l": paired_diff(b["rl"], r["rl"])}
        by_t, by_d = {}, {}
        for key, out in (("q_type", by_t), ("dimension", by_d)):
            groups = defaultdict(list)
            for q in r["js"]:
                groups[refs[q][key]].append(q)
            for g_, qs in sorted(groups.items()):
                out[g_] = paired_diff({q: b["js"][q] for q in qs if q in b["js"]}, {q: r["js"][q] for q in qs})
        imp[(ms, split)]["by_q_type"] = by_t
        imp[(ms, split)]["by_dimension"] = by_d
    with (rd / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "split", "arm", "n_questions", "n_graded", "judge_mean", "judge_lo", "judge_hi", "f1_mean",
                  "f1_lo", "f1_hi", "rougeL_mean", "rougeL_lo", "rougeL_hi", "delta_judge", "delta_judge_lo",
                  "delta_judge_hi", "delta_f1", "delta_f1_lo", "delta_f1_hi", "delta_rougeL", "delta_rougeL_lo",
                  "delta_rougeL_hi", "epochs", "train_minutes", "peak_vram_gb", "adapter_size_mb", "best_val_loss"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r[k] for k in r}
            if r["arm"] == "finetuned" and (r["model"], r["split"]) in imp:
                i = imp[(r["model"], r["split"])]
                for k, key in (("judge", "judge_accuracy"), ("f1", "token_f1"), ("rougeL", "rouge_l")):
                    row[f"delta_{k}"], row[f"delta_{k}_lo"], row[f"delta_{k}_hi"] = (
                        i[key]["mean"], i[key]["lo"], i[key]["hi"])
                t = train.get(r["model"], {})
                row.update(epochs=t.get("epochs"), train_minutes=t.get("train_minutes"),
                           peak_vram_gb=t.get("peak_vram_gb"), adapter_size_mb=t.get("adapter_size_mb"),
                           best_val_loss=t.get("best_val_loss"))
            w.writerow(row)

    L = [f"# Closed-book comparison: base vs QLoRA fine-tuned ({run})", ""]
    L.append("Judge accuracy: Opus 5.5 grading against the reference answer (correct 1, partial 0.5) on a fixed "
             "stratified subset (same questions for every model and arm). Token F1 and ROUGE-L use all test "
             "questions. Brackets are bootstrap 95% CIs; Δ is fine-tuned minus base, paired bootstrap over the same "
             "questions.\n")
    titles = {"test_indomain": "test_indomain (unseen chunks of training documents)",
              "test_heldout_docs": "test_heldout_docs (unseen documents)",
              "test_seen_facts": "test_seen_facts (new paraphrases of TRAINING questions: facts the fine-tuned model saw)"}
    for split in all_splits:
        L.append(f"## {titles.get(split, split)}\n")
        L.append("| Model | Arm | Judge accuracy | Token F1 | ROUGE-L | n (graded / all) |")
        L.append("|---|---|---|---|---|---|")
        for ms in [m.split("/")[-1].lower() for m in models]:
            for arm in arms:
                r = data.get((ms, arm, split))
                if not r:
                    continue
                o = r["summary"]["overall"]
                L.append(f"| {ms} | {arm} | {fmt(o['judge_accuracy'])} | {fmt(o['token_f1'])} | {fmt(o['rouge_l'])} | "
                         f"{r['summary']['n_graded']} / {r['summary']['n_questions']} |")
            if (ms, split) in imp:
                i = imp[(ms, split)]
                L.append(f"| {ms} | **Δ** | **{fmt(i['judge_accuracy'])}** | {fmt(i['token_f1'])} | {fmt(i['rouge_l'])} | |")
        if split == "test_seen_facts":
            L.append("\nUnsupported claims per answer (judge count vs the reference; this split only):\n")
            L.append("| Model | Arm | 0 | 1 | 2+ |\n|---|---|---|---|---|")
            for ms in [m.split("/")[-1].lower() for m in models]:
                for arm in arms:
                    r = data.get((ms, arm, split))
                    u = (r or {}).get("summary", {}).get("unsupported_claims")
                    if u:
                        n = max(sum(u.values()), 1)
                        L.append(f"| {ms} | {arm} | {u['0']} ({u['0'] / n:.0%}) | {u['1']} ({u['1'] / n:.0%}) | "
                                 f"{u['2+']} ({u['2+'] / n:.0%}) |")
        L.append("")
    L.append("## Improvement in judge accuracy by question type and dimension (Δ, paired 95% CI)\n")
    for (ms, split), i in sorted(imp.items()):
        L.append(f"**{ms}, {split}**\n")
        L.append("| Group | Δ judge accuracy | n |\n|---|---|---|")
        for g_, x in list(i["by_q_type"].items()) + list(i["by_dimension"].items()):
            L.append(f"| {g_} | {fmt(x)} | {x['n']} |")
        L.append("")
    L.append("## Training\n")
    L.append("| Model | Epochs | Steps | Train time (min) | Peak VRAM (GB) | Tokens/s | Best val loss | Adapter (MB) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ms, t in train.items():
        L.append(f"| {ms} | {t['epochs']} | {t['steps']} | {t['train_minutes']} | {t['peak_vram_gb']} | "
                 f"{t['tokens_per_sec']} | {t['best_val_loss']} | {t['adapter_size_mb']} |")
    L.append("\nShared recipe: QLoRA (4-bit NF4, bf16 compute), LoRA r 16 / alpha 32 / dropout 0.05 on all "
             "language-model linear layers, lr 2e-4 cosine, effective batch 16, max length 1,024, seed 42, closed-book "
             "format (answer only), paraphrases as extra examples, each model's own chat template; best epoch by "
             "validation loss.")
    L.append("\n**Post-training quantization:** merged models were not quantized in this run (disk). AWQ and GGUF "
             "exports will be produced for the best model only.")
    (rd / "comparison.md").write_text("\n".join(L) + "\n")
    if rows:
        plot(rd, rows, all_splits)
    environment(rd, cfg, models)
    log.info(f"wrote comparison.md/.csv, judge_accuracy.png, environment.txt in {rd}")
