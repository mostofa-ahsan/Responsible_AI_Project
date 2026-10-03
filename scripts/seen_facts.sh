#!/usr/bin/env bash
# test_seen_facts follow-up (run in tmux "seenrun"): waits for scripts/model_runs.sh and the
# test_seen_facts build to finish, then generates answers for every model x arm (GPU, one at a time),
# grades them (batch path, with unsupported_claims), and rebuilds the comparison report.
# Resumable through the same marker directory as model_runs.sh.
set -u
cd "$(dirname "$0")/.." || exit 1
RUN=run_2026-10-03
PY=.venv/bin/python
MK=results/$RUN/.markers
LOG=logs/seen_facts_run.log
STEPS=logs/model_runs_steps.tsv
MODELS=("Qwen/Qwen3-8B" "google/gemma-4-E4B-it" "meta-llama/Llama-3.1-8B-Instruct")
mkdir -p "$MK"
record() { printf '%s\t%s\t%s\t%s\n' "$(date +%H:%M)" "$1" "$2" "${3:-}" >> "$STEPS"; echo "[$(date +%T)] $1: $2 ${3:-}" | tee -a "$LOG"; }
short() { echo "${1##*/}" | tr '[:upper:]' '[:lower:]'; }
wait_gpu_free() {
  for _ in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && return 0; sleep 30
  done; return 1
}
step() {
  local key=$1 gpu=$2; shift 2
  [ -f "$MK/$key.done" ] && return 0
  if [ "$gpu" = 1 ] && ! wait_gpu_free; then record "$key" FAILED "GPU busy for 2 h"; return 1; fi
  echo "----- [$(date +%T)] $key: $*" >> "$LOG"
  local t0; t0=$(date +%s); "$@" >> "$LOG" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then touch "$MK/$key.done"; record "$key" OK "$(( ($(date +%s) - t0) / 60 )) min"
  else record "$key" FAILED "exit $rc (see $LOG)"; fi
  return $rc
}

echo "===== seen_facts.sh started $(date '+%F %T')" >> "$LOG"
until grep -q "model_runs.sh finished" logs/model_runs.log 2>/dev/null; do sleep 60; done
until grep -q "^\[exit" logs/seen_facts_build.out 2>/dev/null; do sleep 30; done
if ! grep -q "^\[exit 0\]" logs/seen_facts_build.out || [ ! -s data/splits/test_seen_facts.jsonl ]; then
  record seen_build FAILED "see logs/seen_facts_build.out"; exit 1
fi
record seen_build OK "$(wc -l < data/splits/test_seen_facts.jsonl) questions"

for m in "${MODELS[@]}"; do
  s=$(short "$m")
  step "gen_${s}_base_test_seen_facts" 1 $PY src/model_runs.py generate --model "$m" --arm base --split test_seen_facts
  if [ -f "models/$s/adapter/adapter_config.json" ]; then
    step "gen_${s}_finetuned_test_seen_facts" 1 $PY src/model_runs.py generate --model "$m" --arm finetuned --split test_seen_facts
  else
    record "gen_${s}_finetuned_test_seen_facts" SKIPPED "no adapter"
  fi
done
step grade_seen 0 $PY src/seen_facts.py grade
rm -f "$MK/report_seen.done"
step report_seen 0 $PY src/model_runs.py report
echo "===== seen_facts.sh finished $(date '+%F %T')" >> "$LOG"
