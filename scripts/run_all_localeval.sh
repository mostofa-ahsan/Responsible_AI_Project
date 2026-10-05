#!/usr/bin/env bash
# Fully local evaluation of the closed-book fine-tuning experiments (run in tmux session "localeval").
# NO API calls: LLM_OFFLINE=1 and every Python stage refuses to construct an API client (local_common.py).
# Stages are resumable (markers in results/local_eval/.done/), time-boxed, and a failing stage is logged and
# the pipeline CONTINUES. The last stage (paper pack + STATUS.md + commit/push) always runs.
#   S1 judge calibration vs Opus      S2 full judge grading          S3 key-fact decomposition
#   S4 answer-claim decomposition     S5 MiniCheck-7B / Flan-T5 / DeBERTa NLI checks
#   S6 statistics                     S8 optional RAG baseline (gated)   S7 paper pack (always last)
set +e
cd "$(dirname "$0")/.." || exit 1
export LLM_OFFLINE=1
export PIPELINE_START=${PIPELINE_START:-$(date +%s)}
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
LOG=logs/localeval.log
DONE=results/local_eval/.done
ST=results/local_eval/stage_status.tsv
mkdir -p "$DONE" results/local_eval
note() { echo "$(date +%H:%M:%S) [orchestrator] $*" | tee -a "$LOG"; }
echo "===== run_all_localeval.sh started $(date '+%F %T') (pipeline start $(date -d @"$PIPELINE_START" +%T))" >> "$LOG"

# hard guard: an API client must be impossible to construct
if ! $PY - >> "$LOG" 2>&1 <<'EOF'
import sys; sys.path.insert(0, "src")
import local_common, anthropic
try:
    anthropic.Anthropic(api_key="x")
    print("GUARD FAILED: client constructed"); sys.exit(1)
except AssertionError as e:
    print(f"guard ok: {e}")
EOF
then note "API guard check FAILED; refusing to run"; exit 2; fi

# GPU temperature / power: scripts/gpu_watchdog.sh (own tmux session "watchdog") logs every 30 s and creates
# results/.gpu_pause when the GPU runs hot; stages wait for it to clear before starting.
pause_wait() {
  while [ -f results/.gpu_pause ]; do note "GPU pause flag set: waiting 5 min"; sleep 300; done
}

wait_gpu_free() {
  for _ in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && return 0; sleep 10
  done
  note "GPU still busy after 10 min; continuing anyway"
}

# stage NAME BOX_MIN cmd...   (skips if marker exists; deadline exported; hard kill at box + 5 min)
stage() {
  local name=$1 box=$2; shift 2
  if [ -f "$DONE/$name" ]; then note "$name: already done (marker)"; return 0; fi
  wait_gpu_free
  pause_wait
  local t0; t0=$(date +%s)
  export STAGE_DEADLINE=$((t0 + box * 60 - 60))
  note "$name: start (time box ${box} min)"
  timeout --kill-after=60 $((box * 60 + 300)) "$@" >> "$LOG" 2>&1
  local rc=$?
  local mins=$(( ($(date +%s) - t0) / 60 ))
  if [ $rc -eq 0 ]; then
    touch "$DONE/$name"; note "$name: done (${mins} min)"; printf '%s\tdone\t%s\t%s\n' "$name" "$mins" "" >> "$ST"
    bash scripts/git_local_commit.sh "Local evaluation: stage $name done" >> "$LOG" 2>&1
  else
    local why="exit $rc"; [ $rc -eq 124 ] || [ $rc -eq 137 ] && why="time box hit (exit $rc)"
    note "$name: FAILED ($why, ${mins} min); continuing"; printf '%s\tfailed\t%s\t%s\n' "$name" "$mins" "$why" >> "$ST"
  fi
  pkill -f "^[^ ]*python[^ ]* src/vllm_json_worker.py" 2>/dev/null
  return $rc
}

stage S1_calibration 75 $PY src/local_judge.py calibrate
stage S2_grading     60 $PY src/local_judge.py grade
stage S3_keyfacts    30 $PY src/local_facts.py keyfacts
stage S4_claims      45 $PY src/local_facts.py claims
stage S5a_purge_judge 5 $PY src/local_checks.py purge_judge
stage S5b_minicheck7b 50 $PY src/local_checks.py minicheck7b
stage S5c_flant5     15 $PY src/local_checks.py flant5
stage S5d_nli        20 $PY src/local_checks.py nli
stage S6_stats       15 $PY src/local_stats.py
stage S8_rag_optional 5 $PY src/local_checks.py rag_gate
# S7 always runs (no marker skip): paper pack, STATUS.md, then commit + push
rm -f "$DONE/S7_pack"
stage S7_pack        25 $PY src/local_pack.py
$PY src/local_pack.py status >> "$LOG" 2>&1
touch "$DONE/PART1_DONE"
bash scripts/localeval_commit.sh >> "$LOG" 2>&1
note "pipeline finished $(date '+%F %T')"
echo "===== run_all_localeval.sh finished $(date '+%F %T')" >> "$LOG"
