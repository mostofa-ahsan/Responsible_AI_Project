#!/usr/bin/env bash
# Follow-up to run_regen.sh (tmux "regen", window "deploy"): merged 4-bit deployment variants of each model's best
# QA-only epoch on the MATCHING (NF4-dequantized) base, graded and checked, then a final stats + research_materials
# rebuild and push. Starts after run_regen.sh has finished; every stage respects the hard limit of results/regen/deadlines.env.
set +e
cd "$(dirname "$0")/.." || exit 1
export LLM_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTORCH_CUDA_ALLOC_CONF
PY=.venv/bin/python; LOG=logs/regen.log; DONE=results/regen/.done; ST=results/regen/stage_status.tsv; BRANCH=exp/cpt-mixed
source results/regen/deadlines.env
note() { echo "$(date '+%F %T') [regen-deploy] $*" | tee -a "$LOG"; }
until [ -f "$DONE/ALL_DONE" ]; do sleep 60; done
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { note "not on $BRANCH"; exit 2; }
gitlock() { flock .git/regen.lock "$@"; }
commit() { gitlock bash scripts/git_local_commit.sh "$1" >> "$LOG" 2>&1; }
push() { if gitlock env GIT_TERMINAL_PROMPT=0 timeout 180 git push origin "$BRANCH" >> "$LOG" 2>&1; then note "pushed ($(git rev-parse --short HEAD))"; else note "push FAILED"; fi; }
run() {  # run NAME BOX_MIN LIMIT cmd...
  local name=$1 box=$2 limit=$3; shift 3
  [ -f "$DONE/$name" ] && return 0
  local now; now=$(date +%s)
  if [ $((now + box * 60)) -gt "$limit" ]; then printf '%s\tpending\t%s\t%s\n' "$name" "$(date '+%F %T')" "not started: would not finish before $(date -d @$limit +%T)" >> "$ST"; note "$name: PENDING (time)"; return 9; fi
  while [ -f results/.gpu_pause ]; do sleep 300; done
  export STAGE_DEADLINE=$((now + box * 60 - 60)); note "$name: start"
  timeout --kill-after=60 $((box * 60 + 60)) "$@" >> "$LOG" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then touch "$DONE/$name"; printf '%s\tdone\t%s\t\n' "$name" $(( ($(date +%s) - now) / 60 )) >> "$ST"; note "$name: done"; commit "Regen deploy: $name"
  else printf '%s\tfailed\t%s\texit %s\n' "$name" $(( ($(date +%s) - now) / 60 )) "$rc" >> "$ST"; note "$name: FAILED (exit $rc)"; fi
  return $rc
}
LIM=$((HARD - 25 * 60))       # leave 25 min for the final stats + research_materials + push
ok=0
for m in qwen3-8b gemma-4-e4b-it llama-3.1-8b-instruct; do
  run "DEP_build_$m" 14 $((LIM - 33 * 60)) $PY src/regen.py deploy --model $m && ok=1
  $PY src/regen.py purge_dequant --model $m >> "$LOG" 2>&1
done
if [ $ok -eq 1 ]; then
  run DEP_grade 25 $((LIM - 10 * 60)) $PY src/regen.py grade --tier DEP
  run DEP_checks 15 $LIM $PY src/regen.py checks --tier DEP
  if [ -f "$DONE/DEP_checks" ]; then printf 'deployment_variants_matching_base\tdone\t-\tmerged 4-bit of the best QA-only epoch on the NF4-dequantized base (graded + checked)\n' >> "$ST"; fi
fi
rm -f "$DONE/DEP_stats" "$DONE/DEP_materials"
run DEP_stats 10 $((HARD - 12 * 60)) $PY src/regen.py stats
run DEP_materials 10 $((HARD - 8 * 60)) $PY src/build_research_materials.py
$PY src/regen.py status >> "$LOG" 2>&1
commit "Regen: deployment variants on the matching base, final research_materials, STATUS"; push
note "follow-up finished"
