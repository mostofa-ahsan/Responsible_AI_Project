#!/usr/bin/env bash
# 3-epoch QLoRA queue (run in tmux session "epochs"): Llama 3.1 8B -> Qwen3-8B -> Gemma 4 E4B.
# Per model: disk check (>= 10 GB free), then 3x [train one epoch (resumable checkpoint) -> save adapter ->
# vLLM answers + local F1/ROUGE-L for test_indomain, test_heldout_docs, test_seen_facts -> PROGRESS.md].
# NO API calls: LLM_OFFLINE=1 for every step. Resumable: finished epochs/generations are skipped.
# A failing model is logged and the queue moves on. GPU jobs run one at a time.
set -u
cd "$(dirname "$0")/.." || exit 1
export LLM_OFFLINE=1
PY=.venv/bin/python
LOG=logs/epochs_queue.log
DEC=logs/UNATTENDED_DECISIONS.md
QUEUE=("meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B" "google/gemma-4-E4B-it")
echo "===== epochs_queue.sh started $(date '+%F %T')" >> "$LOG"
note() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
decide() { printf '| %s | epochs queue | %s | %s |\n' "$(date +%H:%M)" "$1" "$2" >> "$DEC"; }
short() { echo "${1##*/}" | tr '[:upper:]' '[:lower:]'; }

for m in "${QUEUE[@]}"; do
  s=$(short "$m")
  free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "${free:-0}" -lt 10 ]; then
    note "$s: SKIPPED, only ${free} GB free (< 10 GB)"; decide "skipped $s: ${free} GB free" "disk reserve >= 10 GB"; continue
  fi
  failed=0
  for ep in 1 2 3; do
    adapter=models_epochs/$s/epoch$ep/adapter
    if [ ! -f "$adapter/adapter_config.json" ]; then
      note "$s: training epoch $ep"
      $PY src/train_epochs.py --model "$m" >> "$LOG" 2>&1
      rc=$?
      if [ $rc -ne 0 ] || [ ! -f "$adapter/adapter_config.json" ]; then
        note "$s: epoch $ep training FAILED (exit $rc)"; decide "$s epoch $ep training failed (exit $rc); moved to the next model" "queue continues"
        failed=1; break
      fi
      note "$s: epoch $ep adapter saved"
    fi
    if [ ! -f "results/run_epochs/$s/epoch$ep/test_seen_facts_metrics.json" ]; then
      note "$s: generating answers for epoch $ep"
      if ! $PY src/epochs_eval.py gen --model "$m" --epoch "$ep" >> "$LOG" 2>&1; then
        note "$s: epoch $ep generation FAILED"; decide "$s epoch $ep generation failed; training continues" "answers can be regenerated later from the saved adapter"
      fi
    fi
    $PY src/epochs_eval.py progress >> "$LOG" 2>&1
  done
  [ $failed -eq 0 ] && note "$s: all 3 epochs done"
done
$PY src/epochs_eval.py progress >> "$LOG" 2>&1
echo "===== epochs_queue.sh finished $(date '+%F %T')" >> "$LOG"
