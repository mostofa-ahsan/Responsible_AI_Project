#!/usr/bin/env bash
# Trained-question recall evaluation (tmux session "trained"). NO API calls (LLM_OFFLINE=1; local_common's guard).
# Resumable (markers in results/trained_eval/.done/), time-boxed stages, a failing stage is logged and the pipeline
# continues, local commit after each finished stage, push at the end. Stage order keeps the judge download to ONE:
#   T0 test set  T1 vLLM answers  T2 NF4 answers  T3 delete MiniCheck  T4 AWQ  T5 GGUF (time permitting)
#   T6 judge download + key facts  T7 judge grades (+ seed-43)  T8 delete judge, MiniCheck + NLI checks
#   T9 statistics  T10 paper pack + STATUS (always)
set +e
cd "$(dirname "$0")/.." || exit 1
export LLM_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PIPELINE_START=${PIPELINE_START:-$(date +%s)}
PY=.venv/bin/python
LOG=logs/localeval.log
DONE=results/trained_eval/.done
ST=results/trained_eval/stage_status.tsv
mkdir -p "$DONE"
note() { echo "$(date +%H:%M:%S) [trained] $*" | tee -a "$LOG"; }
echo "===== run_trained_eval.sh started $(date '+%F %T')" >> "$LOG"
if ! $PY - >> "$LOG" 2>&1 <<'EOF'
import sys; sys.path.insert(0, "src")
import local_common, anthropic
try:
    anthropic.Anthropic(api_key="x"); sys.exit(1)
except AssertionError as e:
    print(f"guard ok: {e}")
EOF
then note "API guard check FAILED; refusing to run"; exit 2; fi
tmux has-session -t watchdog 2>/dev/null || tmux new-session -d -s watchdog -n gpu "bash scripts/gpu_watchdog.sh"
pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | tr -d ' ')
note "GPU power limit at start: ${pl} W"
printf 'power_limit_at_start\t%s\t%s\n' "$(date '+%F %T')" "$pl" >> "$ST"

pause_wait() { while [ -f results/.gpu_pause ]; do note "GPU pause flag set: waiting 5 min"; sleep 300; done; }
wait_gpu_free() {
  for _ in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && return 0; sleep 10
  done
}
stage() {   # stage NAME BOX_MIN subcommand
  local name=$1 box=$2 sub=$3
  if [ -f "$DONE/$name" ]; then note "$name: already done (marker)"; return 0; fi
  wait_gpu_free; pause_wait
  local t0; t0=$(date +%s)
  export STAGE_DEADLINE=$((t0 + box * 60 - 60))
  note "$name: start (time box ${box} min)"
  timeout --kill-after=60 $((box * 60 + 300)) $PY src/trained_eval.py "$sub" >> "$LOG" 2>&1
  local rc=$? mins=$(( ($(date +%s) - t0) / 60 ))
  if [ $rc -eq 0 ]; then
    touch "$DONE/$name"; note "$name: done (${mins} min)"; printf '%s\tdone\t%s\t\n' "$name" "$mins" >> "$ST"
    bash scripts/git_local_commit.sh "Trained-question eval: stage $name done" >> "$LOG" 2>&1
  else
    local why="exit $rc"; { [ $rc -eq 124 ] || [ $rc -eq 137 ]; } && why="time box hit (exit $rc)"
    note "$name: FAILED ($why, ${mins} min); continuing"; printf '%s\tfailed\t%s\t%s\n' "$name" "$mins" "$why" >> "$ST"
  fi
  pkill -f "^[^ ]*python[^ ]* src/vllm_(multi_generate|json_worker).py" 2>/dev/null
  return $rc
}

stage T0_testset          5 testset
stage T1_gen_vllm        60 gen_vllm
stage T2_gen_nf4         75 gen_nf4
stage T3_purge_minicheck  5 purge_minicheck
stage T4_awq            120 awq
stage T5_gguf            60 gguf
stage T6_judge_keyfacts  40 judge_keyfacts
stage T7_grade          120 grade
stage T8_checks          60 checks
stage T9_stats           20 stats
rm -f "$DONE/T10_pack"
stage T10_pack           20 pack
$PY src/trained_eval.py status >> "$LOG" 2>&1
bash scripts/localeval_commit.sh "Trained-question recall evaluation: results, deployment variants, paper pack" >> "$LOG" 2>&1
touch "$DONE/ALL_DONE"
note "pipeline finished $(date '+%F %T')"
echo "===== run_trained_eval.sh finished $(date '+%F %T')" >> "$LOG"
