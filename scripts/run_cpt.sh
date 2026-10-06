#!/usr/bin/env bash
# Overnight experiment: arm C = mixed continued pretraining (raw text + QA), all three models (tmux session "cpt").
# NO API calls (LLM_OFFLINE=1; local_common's guard makes API clients impossible to construct).
# Resumable: markers in results/cpt/.done/, incremental outputs; each stage time-boxed; on failure: log, fallback,
# continue. Branch exp/cpt-mixed only (commits refuse on any other branch); push after each model and at the end.
#   C1 concise-base answers   C1b their checks   C2 corpus + leakage audit
#   C3 per model (llama -> qwen -> gemma): train epoch -> answers, x3 (6 h box), checks, INTERIM.md, commit, push
#   C4 judge (download once) + grading   C5 checks after the judge   C6 report + STATUS (always), commit, push
set +e
cd "$(dirname "$0")/.." || exit 1
export LLM_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
LOG=logs/cpt.log
DONE=results/cpt/.done
ST=results/cpt/stage_status.tsv
BRANCH=exp/cpt-mixed
mkdir -p "$DONE" results/cpt
note() { echo "$(date '+%F %T') [cpt] $*" | tee -a "$LOG"; }
echo "===== run_cpt.sh started $(date '+%F %T')" >> "$LOG"
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { note "not on $BRANCH: refusing to run"; exit 2; }
if ! $PY - >> "$LOG" 2>&1 <<'EOF'
import sys; sys.path.insert(0, "src")
import local_common, anthropic
try:
    anthropic.Anthropic(api_key="x"); sys.exit(1)
except AssertionError as e:
    print(f"guard ok: {e}")
EOF
then note "API guard check FAILED; refusing to run"; exit 2; fi
tmux has-session -t watchdog 2>/dev/null || tmux new-session -d -s watchdog -n watchdog "bash scripts/gpu_watchdog.sh"
pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | tr -d ' ')
printf 'power_limit\t%s\t%s\n' "$(date '+%F %T')" "$pl" >> "$ST"
note "GPU power limit at start: ${pl} W"

commit() {   # local commit (this branch only)
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { note "commit skipped: not on $BRANCH"; return; }
  bash scripts/git_local_commit.sh "$1" >> "$LOG" 2>&1
}
push() {
  [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || return
  if GIT_TERMINAL_PROMPT=0 timeout 180 git push origin "$BRANCH" >> "$LOG" 2>&1; then note "pushed $BRANCH ($(git rev-parse --short HEAD))"
  else note "push FAILED (commits stay local)"; printf 'push\tfailed\t%s\t\n' "$(date '+%F %T')" >> "$ST"; fi
}
pause_wait() { while [ -f results/.gpu_pause ]; do note "GPU pause flag set: waiting 5 min"; sleep 300; done; }
wait_gpu_free() {
  for _ in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${n:-0}" -eq 0 ] && return 0; sleep 10
  done
}
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }
stage() {   # stage NAME BOX_MIN cmd...
  local name=$1 box=$2; shift 2
  if [ -f "$DONE/$name" ]; then note "$name: already done (marker)"; return 0; fi
  wait_gpu_free; pause_wait
  local t0; t0=$(date +%s)
  export STAGE_DEADLINE=$((t0 + box * 60 - 60))
  note "$name: start (time box ${box} min; free disk $(free_gb) GB)"
  timeout --kill-after=60 $((box * 60 + 300)) "$@" >> "$LOG" 2>&1
  local rc=$? mins=$(( ($(date +%s) - t0) / 60 ))
  if [ $rc -eq 0 ]; then
    touch "$DONE/$name"; note "$name: done (${mins} min)"; printf '%s\tdone\t%s\t\n' "$name" "$mins" >> "$ST"
    commit "CPT: stage $name done"
  else
    local why="exit $rc"; { [ $rc -eq 124 ] || [ $rc -eq 137 ]; } && why="time box hit (exit $rc)"
    note "$name: FAILED ($why, ${mins} min); continuing"; printf '%s\tfailed\t%s\t%s\n' "$name" "$mins" "$why" >> "$ST"
  fi
  pkill -f "^[^ ]*python[^ ]* src/vllm_(multi_generate|json_worker).py" 2>/dev/null
  return $rc
}
decide() { printf '%s\tdecision\t%s\t%s\n' "$1" "$(date '+%F %T')" "$2" >> "$ST"; note "$1: $2"; }

stage C1_gen_concise 60 $PY src/cpt.py gen_concise
stage C1b_checks_concise 60 $PY src/cpt.py checks --concise-only
if [ ! -f "$DONE/C2_build_mix" ]; then
  if ls data/cpt/{llama-3.1-8b-instruct,qwen3-8b,gemma-4-e4b-it}/epoch3.npz >/dev/null 2>&1 && \
     $PY -c "import json,sys; sys.exit(0 if json.load(open('data/cpt/leakage_audit.json'))['passed'] else 1)"; then
    touch "$DONE/C2_build_mix"; printf 'C2_build_mix\tdone\t0\tbuilt and audited before launch\n' >> "$ST"
  else
    stage C2_build_mix 30 $PY src/build_cpt_mix.py
  fi
fi
[ -f "$DONE/C2_build_mix" ] || { note "corpus build / leakage audit failed: arm C cannot run"; }

# ---------------------------------------------------------------- C3: train arm C, per model
for m in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B google/gemma-4-E4B-it; do
  [ -f "$DONE/C2_build_mix" ] || break
  s=$(echo "${m##*/}" | tr '[:upper:]' '[:lower:]')
  [ -f "$DONE/C3_$s" ] && { note "C3_$s: already done (marker)"; continue; }
  dir=models_cpt/$s/mixC
  # disk: >= 10 GB reserve + ~5 GB for this model (3 adapters + 1 checkpoint); MiniCheck-7B (a Stage 4 swap) goes early
  if [ "$(free_gb)" -lt 15 ] && $PY -c "import sys; sys.path.insert(0,'src'); import local_checks as L; sys.exit(0 if L.minicheck_local() else 1)"; then
    $PY -c "import sys; sys.path.insert(0,'src'); import trained_eval as T, local_checks as L; T.delete_hf_model(L.MC7B, 'C3')" >> "$LOG" 2>&1
    decide "C3_$s" "deleted MiniCheck-7B early (free disk < 15 GB); key-fact recall for later models moves to Stage 4"
  fi
  if [ "$(free_gb)" -lt 15 ]; then decide "C3_$s" "SKIPPED: only $(free_gb) GB free"; continue; fi
  mb=4; [ "$s" = "gemma-4-e4b-it" ] && mb=1
  rank=128; alpha=256
  [ -f "$dir/fallback_rank" ] && { rank=$(cut -d' ' -f1 "$dir/fallback_rank"); alpha=$(cut -d' ' -f2 "$dir/fallback_rank"); }
  [ -f "$dir/fallback_mb" ] && mb=$(cat "$dir/fallback_mb")
  t_model=$(date +%s); [ -f "$dir/box_start" ] && t_model=$(cat "$dir/box_start"); mkdir -p "$dir"; echo "$t_model" > "$dir/box_start"
  box_end=$((t_model + 6 * 3600))
  note "C3_$s: start (micro-batch $mb, rank $rank; box ends $(date -d @$box_end '+%F %T'))"
  for ep in 1 2 3; do
    if [ ! -f "$dir/epoch$ep/adapter/adapter_config.json" ]; then
      now=$(date +%s); left=$((box_end - now))
      est=$($PY -c "import json; e=json.load(open('$dir/epochs.json'))['epochs']; print(int(max(x['minutes'] for x in e)*60*1.05+600))" 2>/dev/null || echo 0)
      if [ "$ep" -gt 1 ] && [ "$left" -lt "$est" ]; then
        decide "C3_$s" "epoch $ep not started: ${left}s left in the 6 h box, epoch needs ~${est}s (finished epochs kept)"; break
      fi
      wait_gpu_free; pause_wait
      tries=0; ok=0
      while [ $tries -lt 3 ]; do
        tries=$((tries + 1))
        note "C3_$s: training epoch $ep (try $tries, micro-batch $mb, rank $rank)"
        # expandable_segments: less fragmentation near the 24 GB limit (training only; it broke vLLM engine start-up)
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --kill-after=60 $((box_end - $(date +%s))) $PY src/train_cpt.py --model "$m" --micro-batch $mb \
          --rank $rank --alpha $alpha >> "$LOG" 2>&1
        rc=$?
        [ -f "$dir/epoch$ep/adapter/adapter_config.json" ] && { ok=1; break; }
        if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then decide "C3_$s" "epoch $ep hit the 6 h box (finished epochs kept)"; break; fi
        if [ $rc -eq 3 ]; then
          if [ $mb -gt 1 ]; then mb=$((mb / 2)); echo $mb > "$dir/fallback_mb"
            decide "C3_$s" "CUDA OOM: micro-batch -> $mb (effective batch stays 16)"
          elif [ $rank -eq 128 ] && [ ! -f "$dir/epochs.json" -o "$ep" -eq 1 ]; then
            rank=64; alpha=128; echo "$rank $alpha" > "$dir/fallback_rank"; rm -rf "$dir/checkpoints" "$dir/epochs.json"
            decide "C3_$s" "CUDA OOM at micro-batch 1: LoRA rank -> r=64, alpha=128 (REPORT IN PAPER)"
          else decide "C3_$s" "CUDA OOM with no fallback left"; break; fi
        else
          ck=$(ls -d $dir/checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
          if [ $tries -eq 2 ] && [ -n "$ck" ]; then rm -rf "$ck"; decide "C3_$s" "removed newest checkpoint $ck after a second failure"; fi
          note "C3_$s: training exit $rc; retrying"
        fi
      done
      [ $ok -eq 1 ] || { decide "C3_$s" "epoch $ep failed; keeping finished epochs"; break; }
      note "C3_$s: epoch $ep adapter saved"
      commit "CPT: $s epoch $ep trained"
    fi
    if [ ! -f "$DONE/gen_${s}_ep$ep" ]; then
      wait_gpu_free; pause_wait
      if timeout 3600 $PY src/cpt.py gen_epoch --model "$s" --epoch $ep >> "$LOG" 2>&1; then
        touch "$DONE/gen_${s}_ep$ep"; note "C3_$s: epoch $ep answers done"
      else decide "C3_$s" "epoch $ep answer generation failed (adapter kept; regenerate later)"; fi
      pkill -f "^[^ ]*python[^ ]* src/vllm_multi_generate.py" 2>/dev/null
    fi
  done
  wait_gpu_free
  timeout 5400 $PY src/cpt.py checks --model "$s" >> "$LOG" 2>&1 || decide "C3_$s" "interim checks failed (Stage 5 retries)"
  $PY src/cpt.py interim >> "$LOG" 2>&1
  rm -rf "$dir/checkpoints"
  touch "$DONE/C3_$s"; printf 'C3_%s\tdone\t%s\t\n' "$s" $(( ($(date +%s) - t_model) / 60 )) >> "$ST"
  commit "CPT: $s arm C finished, INTERIM.md"
  push
done

# ---------------------------------------------------------------- C4/C5: judge once, then checks
stage C4_judge 180 $PY src/cpt.py judge
stage C5_checks_final 120 $PY src/cpt.py checks_final
rm -f "$DONE/C6_report"
stage C6_report 30 $PY src/cpt.py report
$PY src/cpt.py status >> "$LOG" 2>&1
commit "CPT: final report, paper pack, STATUS"
push
touch "$DONE/ALL_DONE"
note "pipeline finished"
echo "===== run_cpt.sh finished $(date '+%F %T')" >> "$LOG"
