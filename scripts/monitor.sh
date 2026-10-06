#!/usr/bin/env bash
cd ~/Responsible_AI_Project
echo "=== $(date '+%H:%M:%S') ==="
echo "--- Queue (latest event) ---"
grep -vE "it/s|s/it|^\s*$" logs/epochs_queue.log | grep -E "^\[" | tail -3
echo "--- Training progress (real ETA from progress bar) ---"
tail -c 3000 logs/epochs_queue.log | tr '\r' '\n' | grep -E "[0-9]+/[0-9]+ \[" | tail -1
grep -E "step [0-9]+/" logs/train_epochs.log | tail -1 | sed 's/.*INFO //' | cut -c1-90
echo "--- GPU / disk ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
df -h --output=avail / | tail -1 | sed 's/^/free disk: /'
echo "--- Per model: epochs done, val loss, F1 (seen / indomain / heldout) ---"
python3 - <<'PY'
import json, os
for m in ["llama-3.1-8b-instruct", "qwen3-8b", "gemma-4-e4b-it"]:
    p = f"models_epochs/{m}/epochs.json"
    eps = json.load(open(p))["epochs"] if os.path.exists(p) else []
    print(f"{m:24s} {len(eps)}/3 epochs")
    for e in eps:
        f1 = []
        for s in ("test_seen_facts", "test_indomain", "test_heldout_docs"):
            q = f"results/run_epochs/{m}/epoch{e['epoch']}/{s}_metrics.json"
            f1.append(f"{json.load(open(q))['token_f1']['mean']:.3f}" if os.path.exists(q) else " ... ")
        print(f"   epoch {e['epoch']}: train {e['train_loss_mean']:.3f}  val {e['val_loss']:.3f}  F1 {' / '.join(f1)}")
PY
