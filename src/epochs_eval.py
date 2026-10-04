"""Per-epoch generation + local metrics for the 3-epoch run, and the PROGRESS.md page. No API calls.

gen      --model M --epoch N   greedy vLLM answers with models_epochs/<m>/epoch<N>/adapter for
                               test_indomain, test_heldout_docs, test_seen_facts ->
                               results/run_epochs/<m>/epoch<N>/<split>_predictions.jsonl, then token F1
                               and ROUGE-L (bootstrap 95% CI, overall and by q_type) -> <split>_metrics.json
progress                       rewrite results/run_epochs/PROGRESS.md from the saved logs and metrics

Usage:
    python src/epochs_eval.py gen --model meta-llama/Llama-3.1-8B-Instruct --epoch 1
    python src/epochs_eval.py progress
"""

import argparse
import json
import time
from datetime import datetime, timedelta

from eval_closedbook import generate_vllm, rouge_l, token_f1, vllm_check
from qa_common import read_jsonl
from run_results import boot
from train_qlora import short_name
from utils import get_logger, load_config, repo_path

QUEUE = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B", "google/gemma-4-E4B-it"]
SPLITS = ["test_indomain", "test_heldout_docs", "test_seen_facts"]
EPOCHS = 3
OUT = "results/run_epochs"


def gen(cfg, model, epoch, log):
    adapter = repo_path("models_epochs") / short_name(model) / f"epoch{epoch}" / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter}")
    ok, why = vllm_check()
    if not ok:
        raise SystemExit(f"vLLM unavailable: {why}")
    out = repo_path(OUT) / short_name(model) / f"epoch{epoch}"
    out.mkdir(parents=True, exist_ok=True)
    system = cfg["train"]["system_prompt"]
    for split in SPLITS:
        rows = read_jsonl(repo_path(cfg["paths"]["splits"]) / f"{split}.jsonl")
        pred_path = out / f"{split}_predictions.jsonl"
        have = {p["qa_id"]: p["prediction"] for p in read_jsonl(pred_path)} if pred_path.exists() else {}
        todo = [r for r in rows if r["qa_id"] not in have]
        t0 = time.time()
        if todo:
            new = generate_vllm(cfg, todo, system, str(adapter), log, model=model)
            with pred_path.open("a", encoding="utf-8") as f:
                for r in todo:
                    if r["qa_id"] in new:
                        f.write(json.dumps({"qa_id": r["qa_id"], "prediction": new[r["qa_id"]]}, ensure_ascii=False) + "\n")
            have.update(new)
        f1 = {r["qa_id"]: token_f1(have[r["qa_id"]], r["answer"]) for r in rows if r["qa_id"] in have}
        rl = {r["qa_id"]: rouge_l(have[r["qa_id"]], r["answer"]) for r in rows if r["qa_id"] in have}
        by_type = {}
        for t in sorted({r["q_type"] for r in rows}):
            qs = [r["qa_id"] for r in rows if r["q_type"] == t and r["qa_id"] in have]
            by_type[t] = {"token_f1": boot([f1[q] for q in qs]), "rouge_l": boot([rl[q] for q in qs])}
        metrics = {"model": model, "epoch": epoch, "split": split, "n": len(f1), "n_questions": len(rows),
                   "token_f1": boot(list(f1.values())), "rouge_l": boot(list(rl.values())),
                   "mean_answer_words": round(sum(len(have[q].split()) for q in f1) / max(len(f1), 1), 1),
                   "by_q_type": by_type, "generation_seconds": round(time.time() - t0),
                   "decoding": "greedy", "max_new_tokens": cfg["eval"]["max_new_tokens"], "backend": "vllm+lora"}
        (out / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
        log.info(f"{short_name(model)} epoch {epoch} {split}: F1 {metrics['token_f1']['mean']}, "
                 f"ROUGE-L {metrics['rouge_l']['mean']} ({len(f1)} answers)")


def progress(cfg):
    lines = ["# 3-epoch QLoRA run: progress", "",
             f"_Updated {datetime.now():%Y-%m-%d %H:%M}. Same recipe as run_2026-10-03, except 3 epochs "
             "(cosine over all 3). No API calls (LLM_OFFLINE=1); judge grading comes later._", ""]
    epoch_minutes, remaining = [], 0
    gen_minutes = []
    for model in QUEUE:
        s = short_name(model)
        state_p = repo_path("models_epochs") / s / "epochs.json"
        state = json.loads(state_p.read_text()) if state_p.exists() else {"epochs": []}
        done = state["epochs"]
        remaining += EPOCHS - len(done)
        epoch_minutes += [e["minutes"] for e in done]
        lines.append(f"## {s}: {len(done)}/{EPOCHS} epochs done")
        if not done:
            lines.append("_not started_\n")
            continue
        lines.append("\n| Epoch | Train loss (last / mean) | Val loss | Minutes | Peak VRAM (GB) | "
                     + " | ".join(f"{sp} F1 / ROUGE-L" for sp in SPLITS) + " |")
        lines.append("|---|---|---|---|---|" + "---|" * len(SPLITS))
        for e in done:
            cells = []
            for sp in SPLITS:
                mp = repo_path(OUT) / s / f"epoch{e['epoch']}" / f"{sp}_metrics.json"
                if mp.exists():
                    m = json.loads(mp.read_text())
                    cells.append(f"{m['token_f1']['mean']:.3f} / {m['rouge_l']['mean']:.3f}")
                    gen_minutes.append(m.get("generation_seconds", 0) / 60)
                else:
                    cells.append("pending")
            lines.append(f"| {e['epoch']} | {e['train_loss_last']} / {e['train_loss_mean']} | {e['val_loss']} | "
                         f"{e['minutes']} | {e['peak_vram_gb']} | " + " | ".join(cells) + " |")
        lines.append("")
    per_epoch = (sum(epoch_minutes) / len(epoch_minutes) if epoch_minutes else 70) + \
        (sum(gen_minutes) / max(len(gen_minutes), 1) * len(SPLITS) + 3 if gen_minutes else 8)
    eta = datetime.now() + timedelta(minutes=remaining * per_epoch)
    lines.append(f"**Queue ETA:** {remaining} epoch(s) left, ~{per_epoch:.0f} min each (training + 3 generations) "
                 f"-> about {eta:%H:%M}" + (" (tomorrow)" if eta.date() > datetime.now().date() else "") + ".")
    lines.append("\nAdapters: `models_epochs/<model>/epoch{1,2,3}/adapter`; logs: `models_epochs/<model>/"
                 "{train_loss.csv,epochs.json}`; answers and metrics: `results/run_epochs/<model>/epoch<N>/`.")
    p = repo_path(OUT) / "PROGRESS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["gen", "progress"])
    ap.add_argument("--model")
    ap.add_argument("--epoch", type=int)
    args = ap.parse_args()
    cfg = load_config()
    log = get_logger("epochs_eval", cfg)
    if args.cmd == "gen":
        gen(cfg, args.model, args.epoch, log)
    progress(cfg)


if __name__ == "__main__":
    main()
