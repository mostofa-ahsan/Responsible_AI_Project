#!/usr/bin/env bash
# Runs everything after the full_v1 filter stage, unattended (start it in tmux session "after"):
#   wait for the filter -> run_report -> split (leakage + dimension checks) -> review export (+ copy
#   to Downloads) -> train_qlora dry run -> 30-step GPU smoke test -> base closed-book eval (vLLM,
#   graded within budget) -> logs/READY_FOR_TRAINING.md
# Each step's result goes to logs/after_filter_steps.tsv (time, step, status, detail); full output
# to logs/after_filter.log. Non-critical failures are logged and skipped; steps that need a failed
# critical step are marked SKIPPED. GPU steps wait until the GPU has no other compute process.
set -u
cd "$(dirname "$0")/.." || exit 1
RUN=${RUN:-full_v1}
PY=.venv/bin/python
LOG=logs/after_filter.log
STEPS=logs/after_filter_steps.tsv
DOWNLOADS=/mnt/c/Users/AHSAN-PC/Downloads
: > "$STEPS"
echo "===== after_filter.sh started $(date '+%F %T') (run $RUN)" >> "$LOG"

record() { printf '%s\t%s\t%s\t%s\n' "$(date +%H:%M)" "$1" "$2" "${3:-}" >> "$STEPS"; echo "[$(date +%T)] $1: $2 ${3:-}" | tee -a "$LOG"; }

step() {   # step NAME CMD...   -> returns the command's exit code
  local name=$1; shift
  echo "----- [$(date +%T)] $name: $*" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then record "$name" OK; else record "$name" FAILED "exit $rc (see $LOG)"; fi
  return $rc
}

wait_gpu_free() {
  for _ in $(seq 1 240); do   # up to 2 h
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && return 0
    sleep 30
  done
  return 1
}

# 0. wait for the filter stage (the orchestrator writes state=done/error/stopped_billing)
STATUS=logs/full_run_${RUN}_status.json
while true; do
  state=$($PY -c "import json;print(json.load(open('$STATUS')).get('state',''))" 2>/dev/null)
  case "$state" in done|error|stopped_billing) break;; esac
  sleep 60
done
# the qa tmux chain builds the review workbook right after the run; let it finish first
while pgrep -f "^[^ ]*python[^ ]* src/build_review_xlsx.py --run $RUN" > /dev/null; do sleep 10; done   # anchored: the tmux server argv also contains this text
if [ "$state" != "done" ]; then
  record "filter stage" FAILED "state=$state; dataset steps skipped"
  step "write READY_FOR_TRAINING.md" $PY src/write_ready.py --run "$RUN"
  exit 1
fi
record "filter stage" OK "state=done"

# 1. final report
step "run_report" $PY src/run_report.py --run "$RUN"

# 2. splits (critical for training and eval)
SPLIT_OK=0
if step "split.py" $PY src/split.py --run "$RUN" --max-tries 5; then
  SPLIT_OK=1
  missing=$($PY -c "import json;m=json.load(open('data/splits/split_manifest.json'));print(sum(len(v) for v in m['missing_dimensions'].values()))")
  [ "$missing" != "0" ] && record "split dimension coverage" WARN "$missing dimension gaps across test sets after re-seeding (logged)"
fi

# 3. review export + copy to Downloads
if step "review export" $PY src/build_review_xlsx.py --run "$RUN" --layout v3 --n-passed 100 --n-rejected 30 \
     --out "data/qa_pairs/${RUN}_review.xlsx"; then
  if cp "data/qa_pairs/${RUN}_review.xlsx" "$DOWNLOADS/"; then record "copy review to Downloads" OK "$DOWNLOADS/${RUN}_review.xlsx"
  else record "copy review to Downloads" FAILED; fi
fi

if [ $SPLIT_OK -eq 1 ]; then
  # 4. training dry run (CPU)
  step "train_qlora --dry-run" env CUDA_VISIBLE_DEVICES="" $PY src/train_qlora.py --dry-run

  # 5. 30-step GPU smoke test, then delete its checkpoints
  if wait_gpu_free; then
    step "smoke test (30 steps)" $PY src/train_qlora.py --max-steps 30 --output models/smoke
    rm -rf models/smoke && record "delete smoke checkpoint" OK
  else
    record "smoke test (30 steps)" SKIPPED "GPU busy for 2 h"
  fi

  # 6. base-model closed-book eval (vLLM generation on GPU, Opus grading via batch within budget)
  for split in test_indomain test_heldout_docs; do
    if wait_gpu_free; then
      step "base eval $split" $PY src/eval_closedbook.py --arm base_closedbook --split "$split"
    else
      record "base eval $split" SKIPPED "GPU busy for 2 h"
    fi
  done
else
  for s in "train_qlora --dry-run" "smoke test (30 steps)" "base eval test_indomain" "base eval test_heldout_docs"; do
    record "$s" SKIPPED "split.py failed"
  done
fi

# 7. summary page
step "write READY_FOR_TRAINING.md" $PY src/write_ready.py --run "$RUN"
echo "===== after_filter.sh finished $(date '+%F %T')" >> "$LOG"
