"""Seed replication (seed 43) of the 3-epoch run's epoch-1 adapters: answers, local metrics, summary. No API calls.

gen     --model M   greedy vLLM answers (same settings as run_epochs) with models_seeds/<m>/seed43/epoch1/adapter on
                    the eval-subset ids of all 3 test splits -> results/seed_replication/<m>/seed43_epoch1/<split>_predictions.jsonl
checks  --model M   key-fact support (primary checker + Flan-T5) and DeBERTa NLI contradiction against the CACHED
                    key facts (data/eval_keyfacts/) -> results/seed_replication/<m>/seed43_epoch1/*.jsonl
summary             seed 42 vs seed 43 at epoch 1 per model and split: token F1, ROUGE-L, key-fact recall,
                    contradiction rate (paired bootstrap 95% CIs on identical items) and val loss
                    -> results/seed_replication/SUMMARY.md, summary.csv. Judge grades for seed 43: pending (no judge weights).
Every step is resumable and writes incrementally.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from local_common import KEYFACTS, OUT, SPLITS, FAMILY_LABEL, items, log, pause_wait, read_jsonl, write_jsonl
from utils import load_config, repo_path

SEED = 43
SR = repo_path("results/seed_replication")
MODELS = {"gemma-4-e4b-it": "google/gemma-4-E4B-it", "qwen3-8b": "Qwen/Qwen3-8B",
          "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct"}


def short(model):
    return model.split("/")[-1].lower()


def out_dir(fam):
    d = SR / fam / f"seed{SEED}_epoch1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def adapter(fam):
    return repo_path("models_seeds") / fam / f"seed{SEED}" / "epoch1" / "adapter"


def sid(fam):
    return f"{fam}__seed{SEED}_ep1"


def seed_preds(fam):
    d = out_dir(fam)
    return {s: {r["qa_id"]: r["prediction"] for r in read_jsonl(d / f"{s}_predictions.jsonl")} for s in SPLITS}


def cmd_gen(model):
    from eval_closedbook import generate_vllm
    cfg = load_config()
    fam = short(model)
    if not (adapter(fam) / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter(fam)}")
    its = items()
    have = seed_preds(fam)
    todo = [dict(r, _split=s) for s in SPLITS for q, r in its[s].items() if q not in have[s]]
    log().info(f"[seeds] {fam}: generating {len(todo)} answers (eval subset, 3 splits)")
    if todo:
        pause_wait("seeds gen")
        t0 = time.time()
        new = generate_vllm(cfg, todo, cfg["train"]["system_prompt"], str(adapter(fam)), log(), model=model)
        for s in SPLITS:
            with (out_dir(fam) / f"{s}_predictions.jsonl").open("a", encoding="utf-8") as f:
                for r in todo:
                    if r["_split"] == s and r["qa_id"] in new:
                        f.write(json.dumps({"qa_id": r["qa_id"], "prediction": new[r["qa_id"]]}, ensure_ascii=False) + "\n")
        log().info(f"[seeds] {fam}: {len(new)} answers in {(time.time() - t0) / 60:.1f} min")
    got = sum(len(v) for v in seed_preds(fam).values())
    if got < sum(len(v) for v in its.values()):
        raise SystemExit(f"{fam}: only {got} answers")


def fact_pairs(fam):
    kf = {s: {r["qa_id"]: r["facts"] for r in read_jsonl(KEYFACTS / f"{s}.jsonl")} for s in SPLITS}
    p = seed_preds(fam)
    return [(f"{s}|{q}|{sid(fam)}|{k}", s, q, sid(fam), k, p[s][q], f)
            for s in SPLITS for q in p[s] if q in kf[s] for k, f in enumerate(kf[s][q])]


def primary_checker():
    v = OUT / "validation.json"
    if v.exists():
        return json.loads(v.read_text()).get("meta", {}).get("primary_checker", "flant5")
    n = len(read_jsonl(OUT / "per_item" / "support_minicheck7b.jsonl"))
    return "minicheck7b" if n > 0.9 * len(read_jsonl(OUT / "per_item" / "support_flant5.jsonl")) else "flant5"


def cmd_checks(model):
    import local_checks as L
    fam = short(model)
    d = out_dir(fam)
    pairs = fact_pairs(fam)
    far = time.time() + 3 * 3600
    log().info(f"[seeds] {fam}: checks on {len(pairs)} (answer, fact) pairs; primary checker {primary_checker()}")
    if primary_checker() == "minicheck7b":
        pause_wait("seeds minicheck")
        L.score_minicheck7b(pairs, d / "minicheck7b_raw.jsonl", d / "support_minicheck7b.jsonl", far, "seeds MiniCheck-7B")
    pause_wait("seeds flant5")
    L.score_flant5(pairs, d / "support_flant5.jsonl", far, "seeds Flan-T5")
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    pause_wait("seeds nli")
    L.score_nli_facts(L.NLIScorer(), pairs, d / "nli_facts.jsonl", far, "seeds NLI")


def per_item_seed43(fam, prim):
    from eval_closedbook import rouge_l, token_f1
    its = items()
    d = out_dir(fam)
    kf = {(s, r["qa_id"]): len(r["facts"]) for s in SPLITS for r in read_jsonl(KEYFACTS / f"{s}.jsonl")}

    def load(name, field):
        out = {}
        for r in read_jsonl(d / name):
            out.setdefault((r["split"], r["qa_id"]), {})[r["k"]] = r[field]
        return out
    sup = load(f"support_{prim}.jsonl", "p")
    con = load("nli_facts.jsonl", "contradicted")
    rows = []
    for s, ps in seed_preds(fam).items():
        for q, a in ps.items():
            n = kf.get((s, q))
            sp, cn = sup.get((s, q), {}), con.get((s, q), {})
            rows.append({"split": s, "qa_id": q, "token_f1": token_f1(a, its[s][q]["answer"]),
                         "rouge_l": rouge_l(a, its[s][q]["answer"]),
                         "keyfact_recall": sum(v >= 0.5 for v in sp.values()) / n if n and len(sp) >= n else np.nan,
                         "contradiction_rate": sum(cn.values()) / n if n and len(cn) >= n else np.nan,
                         "answer_words": len(a.split())})
    return pd.DataFrame(rows)


def cmd_summary():
    from local_stats import boot_ci, paired_boot
    mi = OUT / "master_items.csv"
    if not mi.exists():
        raise SystemExit("results/local_eval/master_items.csv missing (S6 not done)")
    master = pd.read_csv(mi)
    prim = primary_checker()
    metrics = [("token_f1", "Token F1"), ("rouge_l", "ROUGE-L"), ("keyfact_recall", f"Key-fact recall ({prim})"),
               ("contradiction_rate", "Contradiction rate"), ("answer_words", "Mean answer words")]
    out, lines = [], []
    for fam, model in MODELS.items():
        if not seed_preds(fam)["test_seen_facts"]:
            continue
        s43 = per_item_seed43(fam, prim)
        s42 = master[master.system == f"{fam}__ep1"]
        st42 = json.loads((repo_path("models_epochs") / fam / "epochs.json").read_text())["epochs"][0]
        p43 = repo_path("models_seeds") / fam / f"seed{SEED}" / "epochs.json"
        st43 = json.loads(p43.read_text())["epochs"][0] if p43.exists() else {}
        for s in SPLITS:
            a = s42[s42.split == s].set_index("qa_id")
            b = s43[s43.split == s].set_index("qa_id")
            common = a.index.intersection(b.index)
            for col, lab in metrics:
                x42 = a.loc[common, col].to_numpy(dtype=float)
                x43 = b.loc[common, col].to_numpy(dtype=float)
                m42, m43 = boot_ci(x42.tolist()), boot_ci(x43.tolist())
                diff = paired_boot(x43 - x42)
                out.append({"family": fam, "split": s, "metric": col, "seed42": m42[0], "seed42_lo": m42[1],
                            "seed42_hi": m42[2], "seed43": m43[0], "seed43_lo": m43[1], "seed43_hi": m43[2],
                            "diff_43_minus_42": diff["diff"], "abs_diff": abs(diff["diff"]) if diff["diff"] == diff["diff"] else np.nan,
                            "diff_lo": diff["lo"], "diff_hi": diff["hi"], "p": diff["p"], "n": diff["n"]})
        out.append({"family": fam, "split": "-", "metric": "val_loss", "seed42": st42.get("val_loss"),
                    "seed43": st43.get("val_loss"),
                    "diff_43_minus_42": (st43["val_loss"] - st42["val_loss"]) if st43 else np.nan,
                    "abs_diff": abs(st43["val_loss"] - st42["val_loss"]) if st43 else np.nan})
    df = pd.DataFrame(out)
    SR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SR / "summary.csv", index=False)

    def f(x, d=3):
        return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
    L = ["# Seed replication: epoch 1, seed 42 vs seed 43", "",
         f"_Generated by `src/seed_eval.py summary` at {time.strftime('%Y-%m-%d %H:%M')}; numbers from `summary.csv`._", "",
         "Same recipe as `models_epochs` epoch 1 (QLoRA, cosine schedule over 3 epochs, stopped after epoch 1); only "
         "the seed (LoRA initialisation and data order) differs. Eval-subset items (500 / 500 / 279), greedy vLLM "
         f"decoding. Key-fact recall uses the cached key facts and the primary checker ({prim}); contradiction = "
         "DeBERTa-v3-large NLI. Δ = seed 43 − seed 42 with a paired bootstrap 95% CI on identical items. "
         "**Judge grades for seed 43: pending** (the local judge weights were deleted to make room for MiniCheck-7B).", ""]
    if df.empty:
        L.append("_No model finished yet._")
    for fam in MODELS:
        g = df[df.family == fam]
        if g.empty:
            L += [f"## {FAMILY_LABEL[fam]}", "", "_pending_", ""]
            continue
        L += [f"## {FAMILY_LABEL[fam]}", "", "| Split | Metric | Seed 42 | Seed 43 | Δ (43 − 42) [95% CI] | |Δ| |",
              "|---|---|---|---|---|---|"]
        for r in g.itertuples():
            if r.metric == "val_loss":
                L.append(f"| – | Validation loss | {f(r.seed42)} | {f(r.seed43)} | {f(r.diff_43_minus_42)} | {f(r.abs_diff)} |")
            else:
                ci = f"[{f(r.diff_lo)}, {f(r.diff_hi)}]"
                L.append(f"| {r.split} | {dict(metrics)[r.metric]} | {f(r.seed42)} | {f(r.seed43)} | "
                         f"{f(r.diff_43_minus_42)} {ci} | {f(r.abs_diff)} |")
        L.append("")
    m = df[(df.metric != "val_loss") & (df.metric != "answer_words")]
    if not m.empty:
        excl = int(((m.diff_lo > 1e-9) | (m.diff_hi < -1e-9)).sum())     # tolerance: CSV round-trip noise
        L.append(f"Across finished models: {excl} of {len(m)} metric × split differences have a 95% CI excluding 0; "
                 f"largest |Δ|: " + ", ".join(f"{dict(metrics)[c]} {m[m.metric == c].abs_diff.max():.3f}"
                                              for c in ("token_f1", "rouge_l", "keyfact_recall", "contradiction_rate")
                                              if m[m.metric == c].abs_diff.notna().any()) + ".")
    (SR / "SUMMARY.md").write_text("\n".join(L) + "\n")
    log().info(f"[seeds] summary written ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "checks", "summary"])
    ap.add_argument("--model")
    a = ap.parse_args()
    if a.cmd == "summary":
        cmd_summary()
    else:
        {"gen": cmd_gen, "checks": cmd_checks}[a.cmd](a.model)


if __name__ == "__main__":
    main()
