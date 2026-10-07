#!/usr/bin/env bash
# Serve adapters on their training base (NF4-dequantized bf16) + research_materials/ (tmux session "regen").
# NO API calls (LLM_OFFLINE=1; local_common's guard). Branch exp/cpt-mixed only. Resumable (results/regen/.done/),
# time-boxed stages, fallbacks, local commit after each stage.
# Deadlines (fixed at the first launch, kept in results/regen/deadlines.env):
#   CHECKPOINT = start + 2 h 55 min: P1-P2 graded + checked and a first research_materials/ build committed and pushed
#   HARD       = start + 6 h: everything committed and pushed (final build rerun after the last stage)
set +e
cd "$(dirname "$0")/.." || exit 1
export LLM_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF
PY=.venv/bin/python
LOG=logs/regen.log
DONE=results/regen/.done
ST=results/regen/stage_status.tsv
BRANCH=exp/cpt-mixed
mkdir -p "$DONE" results/regen
DL=results/regen/deadlines.env
if [ ! -f "$DL" ]; then
  s=$(date +%s); printf 'START=%s\nCHECKPOINT=%s\nHARD=%s\n' "$s" $((s + 175 * 60)) $((s + 360 * 60)) > "$DL"
fi
source "$DL"
note() { echo "$(date '+%F %T') [regen] $*" | tee -a "$LOG"; }
echo "===== run_regen.sh started $(date '+%F %T') (checkpoint $(date -d @$CHECKPOINT +%T), hard $(date -d @$HARD +%T))" >> "$LOG"
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { note "not on $BRANCH: refusing to run"; exit 2; }
if ! $PY - >> "$LOG" 2>&1 <<'EOF'
import sys; sys.path.insert(0, "src")
import local_common, anthropic
try:
    anthropic.Anthropic(api_key="x"); sys.exit(1)
except AssertionError as e:
    print(f"guard ok: {e}")
EOF
then note "API guard check FAILED"; exit 2; fi
tmux has-session -t watchdog 2>/dev/null || tmux new-session -d -s watchdog -n watchdog "bash scripts/gpu_watchdog.sh"
printf 'power_limit\t%s\t%s\n' "$(date '+%F %T')" "$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | tr -d ' ')" >> "$ST"

gitlock() { flock .git/regen.lock "$@"; }
commit() { [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] && gitlock bash scripts/git_local_commit.sh "$1" >> "$LOG" 2>&1; }
push() {
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || return
  if gitlock env GIT_TERMINAL_PROMPT=0 timeout 180 git push origin "$BRANCH" >> "$LOG" 2>&1; then note "pushed $BRANCH ($(git rev-parse --short HEAD))"
  else note "push FAILED"; printf 'push\tfailed\t%s\t\n' "$(date '+%F %T')" >> "$ST"; fi
}
# safety net: commit + push at CHECKPOINT - 10 min and HARD - 10 min, whatever is running
( for t in $((CHECKPOINT - 600)) $((HARD - 600)); do
    while [ "$(date +%s)" -lt "$t" ]; do sleep 30; done
    $PY src/regen.py status >> "$LOG" 2>&1; commit "Regen: timed safety commit"; push
  done ) &
pause_wait() { while [ -f results/.gpu_pause ]; do note "GPU pause flag set: waiting 5 min"; sleep 300; done; }
wait_gpu_free() {
  for _ in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && return 0; sleep 10
  done
}
pending() { printf '%s\tpending\t%s\t%s\n' "$1" "$(date '+%F %T')" "$2" >> "$ST"; note "$1: PENDING ($2)"; }
stage() {   # stage NAME BOX_MIN LIMIT_EPOCH cmd...   (not started if it cannot finish before LIMIT)
  local name=$1 box=$2 limit=$3; shift 3
  if [ -f "$DONE/$name" ]; then note "$name: already done (marker)"; return 0; fi
  local now; now=$(date +%s)
  if [ $((now + 60)) -ge "$limit" ]; then pending "$name" "no time before $(date -d @$limit +%T)"; return 9; fi
  [ $((now + box * 60)) -gt "$limit" ] && box=$(( (limit - now) / 60 ))
  wait_gpu_free; pause_wait
  export STAGE_DEADLINE=$((now + box * 60 - 60))
  note "$name: start (box ${box} min; free disk $(df -BG --output=avail . | tail -1 | tr -dc '0-9') GB)"
  timeout --kill-after=60 $((box * 60 + 120)) "$@" >> "$LOG" 2>&1
  local rc=$? mins=$(( ($(date +%s) - now) / 60 ))
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; note "$name: done (${mins} min)"; printf '%s\tdone\t%s\t\n' "$name" "$mins" >> "$ST"
    commit "Regen: stage $name done"
  else local why="exit $rc"; { [ $rc -eq 124 ] || [ $rc -eq 137 ]; } && why="time box hit (exit $rc)"
    note "$name: FAILED ($why, ${mins} min); continuing"; printf '%s\tfailed\t%s\t%s\n' "$name" "$mins" "$why" >> "$ST"; fi
  pkill -f "^[^ ]*python[^ ]* src/vllm_(multi_generate|json_worker).py" 2>/dev/null
  return $rc
}
PRE=$((CHECKPOINT - 35 * 60))      # pre-checkpoint work must leave 35 min for stats + research_materials + push

# ---- Part A, generation: one dequantized base at a time (16 GB each; MiniCheck-7B removed first for disk)
if [ ! -f "$DONE/R1_gemma-4-e4b-it" ] || [ ! -f "$DONE/R1_llama-3.1-8b-instruct" ] || [ ! -f "$DONE/R1_qwen3-8b" ]; then
  $PY -c "import sys; sys.path.insert(0,'src'); import local_checks as L, trained_eval as T; L.minicheck_local() and T.delete_hf_model(L.MC7B, 'R1')" >> "$LOG" 2>&1
fi
for m in qwen3-8b gemma-4-e4b-it llama-3.1-8b-instruct; do
  [ -f "$DONE/R1_$m" ] && continue
  if stage "R1_dequant_$m" 20 $PRE $PY src/regen.py dequant --model $m; then
    stage "R1_gen_$m" 30 $PRE $PY src/regen.py gen --model $m && touch "$DONE/R1_$m"
  else pending "R1_$m" "dequantized base not built/verified: python src/regen.py dequant --model $m"; fi
  $PY src/regen.py purge_dequant --model $m >> "$LOG" 2>&1
done
# ---- grade + check P1-P2 before the checkpoint
stage R_grade_P12 90 $((PRE - 20 * 60)) $PY src/regen.py grade --tier P12
stage R_checks_P12 30 $PRE $PY src/regen.py checks --tier P12
# ---- checkpoint: stats + first research_materials build, commit, push
rm -f "$DONE/R_stats_1" "$DONE/R_materials_1"
stage R_stats_1 15 $((CHECKPOINT - 5 * 60)) $PY src/regen.py stats
stage R_materials_1 20 $((CHECKPOINT - 3 * 60)) $PY src/build_research_materials.py
$PY src/regen.py status >> "$LOG" 2>&1
commit "Regen: checkpoint (P1-P2 on the matching base, first research_materials build)"; push
note "checkpoint reached"
# ---- after the checkpoint: P3, P4, deferred; final build before HARD
FIN=$((HARD - 30 * 60))
stage R_grade_rest 120 $((FIN - 45 * 60)) $PY src/regen.py grade --tier rest
stage R_checks_rest 45 $((FIN - 5 * 60)) $PY src/regen.py checks --tier rest
pending "deployment_variants_matching_base" "merged AWQ/RTN/GGUF on the dequantized base need the dequantized checkpoint + llm-compressor + a further judge/MiniCheck swap; resume: see STATUS.md"
rm -f "$DONE/R_stats_final" "$DONE/R_materials_final"
stage R_stats_final 15 $((HARD - 15 * 60)) $PY src/regen.py stats
stage R_materials_final 20 $((HARD - 12 * 60)) $PY src/build_research_materials.py
$PY src/regen.py status >> "$LOG" 2>&1
commit "Regen: final results, research_materials, STATUS"; push
touch "$DONE/ALL_DONE"
note "pipeline finished"
echo "===== run_regen.sh finished $(date '+%F %T')" >> "$LOG"
