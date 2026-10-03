#!/usr/bin/env bash
# Multi-model QLoRA fine-tuning + closed-book evaluation (run in tmux session "models").
# Resumable: every finished step leaves results/<run>/.markers/<step>.done and is skipped on re-run.
# Order: dataset check -> eval subset -> downloads -> smoke tests (all models) -> epoch choice ->
#        per model: train -> generate base/finetuned x 2 splits -> grading (batch) -> report.
# GPU steps run strictly one at a time. Results of each step: logs/model_runs_steps.tsv; full log:
# logs/model_runs.log; decisions: logs/UNATTENDED_DECISIONS.md.
set -u
cd "$(dirname "$0")/.." || exit 1
RUN=run_2026-10-03
PY=.venv/bin/python
R=results/$RUN
MK=$R/.markers
LOG=logs/model_runs.log
STEPS=logs/model_runs_steps.tsv
MODELS=("Qwen/Qwen3-8B" "google/gemma-4-E4B-it" "meta-llama/Llama-3.1-8B-Instruct")
SPLITS=(test_indomain test_heldout_docs)
mkdir -p "$MK"
echo "===== model_runs.sh started $(date '+%F %T')" >> "$LOG"

record() { printf '%s\t%s\t%s\t%s\n' "$(date +%H:%M)" "$1" "$2" "${3:-}" >> "$STEPS"; echo "[$(date +%T)] $1: $2 ${3:-}" | tee -a "$LOG"; }
short() { echo "${1##*/}" | tr '[:upper:]' '[:lower:]'; }
done_() { [ -f "$MK/$1.done" ]; }

wait_gpu_free() {
  for _ in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && return 0
    sleep 30
  done
  return 1
}

step() {   # step KEY GPU(0|1) CMD...   -> skip if done; record OK/FAILED; return exit code
  local key=$1 gpu=$2; shift 2
  if done_ "$key"; then echo "[$(date +%T)] $key: already done" >> "$LOG"; return 0; fi
  if [ "$gpu" = 1 ] && ! wait_gpu_free; then record "$key" FAILED "GPU busy for 2 h"; return 1; fi
  echo "----- [$(date +%T)] $key: $*" >> "$LOG"
  local t0; t0=$(date +%s)
  "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$MK/$key.done"; record "$key" OK "$(( ($(date +%s) - t0) / 60 )) min"
  else record "$key" FAILED "exit $rc after $(( ($(date +%s) - t0) / 60 )) min (see $LOG)"; fi
  return $rc
}

# 0. dataset steps (already done earlier today; re-run only what is missing)
[ -f logs/full_v1_final_report.txt ] || step dataset_report 0 $PY src/run_report.py --run full_v1
[ -f data/splits/split_manifest.json ] || step dataset_split 0 $PY src/split.py --run full_v1
[ -f /mnt/c/Users/AHSAN-PC/Downloads/full_v1_review.xlsx ] || step dataset_review 0 bash -c \
  "$PY src/build_review_xlsx.py --run full_v1 --layout v3 --n-passed 100 --n-rejected 30 --out data/qa_pairs/full_v1_review.xlsx && cp data/qa_pairs/full_v1_review.xlsx /mnt/c/Users/AHSAN-PC/Downloads/"
[ -f docs/DATASET_CARD_full_v1.md ] && record dataset_card OK "docs/DATASET_CARD_full_v1.md" || record dataset_card MISSING ""

# 1. fixed eval subset
step eval_subset 0 $PY src/model_runs.py subset

# 2. downloads
declare -A OK_MODEL
for m in "${MODELS[@]}"; do
  s=$(short "$m")
  if step "download_$s" 0 $PY src/model_runs.py download --model "$m"; then OK_MODEL[$s]=1; fi
done

# 3. smoke tests for all models, then one epoch count
for m in "${MODELS[@]}"; do
  s=$(short "$m")
  [ -n "${OK_MODEL[$s]:-}" ] || { record "smoke_$s" SKIPPED "no download"; continue; }
  step "smoke_$s" 1 $PY src/model_runs.py smoke --model "$m" || unset "OK_MODEL[$s]"
done
step choose_epochs 0 $PY src/model_runs.py epochs

# 4. per model: train, then generate base and fine-tuned answers on both test splits
for m in "${MODELS[@]}"; do
  s=$(short "$m")
  [ -n "${OK_MODEL[$s]:-}" ] || { record "train_$s" SKIPPED "no download or smoke test failed"; continue; }
  for sp in "${SPLITS[@]}"; do
    step "gen_${s}_base_$sp" 1 $PY src/model_runs.py generate --model "$m" --arm base --split "$sp"
  done
  if step "train_$s" 1 $PY src/model_runs.py train --model "$m"; then
    for sp in "${SPLITS[@]}"; do
      step "gen_${s}_finetuned_$sp" 1 $PY src/model_runs.py generate --model "$m" --arm finetuned --split "$sp"
    done
  else
    record "gen_${s}_finetuned" SKIPPED "training failed"
  fi
done

# 5. grading (Opus via Message Batches, fixed subset, within eval_max_usd), 6. report
step grade 0 $PY src/model_runs.py grade
rm -f "$MK/report.done"            # the report is cheap: always rebuild it from whatever exists
step report 0 $PY src/model_runs.py report
echo "===== model_runs.sh finished $(date '+%F %T')" >> "$LOG"
