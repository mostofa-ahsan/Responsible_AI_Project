#!/usr/bin/env bash
# Seed replication queue (tmux session "seeds"): gemma-4-e4b-it -> qwen3-8b -> llama-3.1-8b-instruct.
# Per model: disk check (>= 10 GB) -> epoch 1 of the 3-epoch recipe with seed 43 (checkpoint every 150 steps,
# resumes mid-epoch after a crash) -> vLLM answers on the eval subset -> key-fact checks (primary checker + Flan-T5 +
# DeBERTa NLI, cached key facts) -> results/seed_replication/SUMMARY.md -> paper pack refresh -> local commit.
# Starts after the local evaluation (results/local_eval/.done/PART1_DONE). Resumable via markers; no API calls.
set +e
cd "$(dirname "$0")/.." || exit 1
export LLM_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
LOG=logs/seeds.log
MK=results/seed_replication/.done
mkdir -p "$MK" results/seed_replication
note() { echo "$(date +%H:%M:%S) [seeds] $*" | tee -a "$LOG" >> logs/localeval.log; }
pause_wait() { while [ -f results/.gpu_pause ]; do note "GPU pause flag set: waiting 5 min"; sleep 300; done; }
wait_gpu_free() {
  for _ in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && return 0; sleep 10
  done
}
step() {   # step KEY cmd...
  local key=$1; shift
  [ -f "$MK/$key" ] && return 0
  wait_gpu_free; pause_wait
  local t0; t0=$(date +%s)
  note "$key: start"
  "$@" >> "$LOG" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then touch "$MK/$key"; note "$key: done ($(( ($(date +%s) - t0) / 60 )) min)"
  else note "$key: FAILED (exit $rc)"; fi
  return $rc
}

echo "===== seeds_queue.sh started $(date '+%F %T')" >> "$LOG"
until [ -f results/local_eval/.done/PART1_DONE ]; do sleep 60; done
note "local evaluation finished: starting the seed-43 queue"
for m in google/gemma-4-E4B-it Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct; do
  s=$(echo "${m##*/}" | tr '[:upper:]' '[:lower:]')
  if [ ! -f "$MK/${s}_summary" ]; then
    free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    if [ "${free:-0}" -lt 10 ]; then note "$s: SKIPPED, only ${free} GB free (< 10 GB)"; continue; fi
  fi
  train() {
    $PY src/train_epochs.py --model "$m" --root models_seeds --subdir seed43 --seed 43 --max-epochs 1 --save-steps 150
    local rc=$?; [ $rc -eq 10 ] && return 0; return $rc
  }
  if ! step "${s}_train" train; then
    # a crash can leave the newest checkpoint half-written: drop it and resume from the previous one (once)
    ck=$(ls -d models_seeds/$s/seed43/checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$ck" ]; then note "$s: removing newest checkpoint $ck and retrying"; rm -rf "$ck"; fi
    step "${s}_train" train || { note "$s: training failed twice; next model"; continue; }
  fi
  step "${s}_gen" $PY src/seed_eval.py gen --model "$m" || { note "$s: generation failed; next model"; continue; }
  step "${s}_checks" $PY src/seed_eval.py checks --model "$m" || note "$s: checks failed (summary uses what exists)"
  rm -f "$MK/${s}_summary"
  step "${s}_summary" $PY src/seed_eval.py summary
  $PY src/local_pack.py >> "$LOG" 2>&1
  bash scripts/git_local_commit.sh "Seed replication: $s seed 43 epoch 1" >> "$LOG" 2>&1
done
$PY src/seed_eval.py summary >> "$LOG" 2>&1
$PY src/local_pack.py >> "$LOG" 2>&1
$PY src/local_pack.py status >> "$LOG" 2>&1
bash scripts/localeval_commit.sh "Seed replication (seed 43, epoch 1) and refreshed paper pack" >> "$LOG" 2>&1
note "seed queue finished $(date '+%F %T')"
echo "===== seeds_queue.sh finished $(date '+%F %T')" >> "$LOG"
