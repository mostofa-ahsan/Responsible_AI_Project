#!/usr/bin/env bash
# After a crash / reboot: relaunch the matching-base regeneration + research_materials pipeline (tmux "regen").
# Finished stages are skipped (results/regen/.done/); deadlines stay those of the first launch
# (results/regen/deadlines.env). Check the GPU power cap in Windows first.
#   bash scripts/resume_regen_after_crash.sh
cd "$(dirname "$0")/.." || exit 1
[ "$(git rev-parse --abbrev-ref HEAD)" = "exp/cpt-mixed" ] || { echo "switch to branch exp/cpt-mixed first"; exit 1; }
for f in logs/regen.log logs/regen_worker.log logs/localeval.log; do
  [ -f "$f" ] && tr -d '\000' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
rm -f results/regen/raw/*.in.jsonl
tmux has-session -t watchdog 2>/dev/null || tmux new-session -d -s watchdog -n watchdog "bash scripts/gpu_watchdog.sh"
[ -f results/regen/.done/ALL_DONE ] && { echo "already finished"; exit 0; }
tmux has-session -t regen 2>/dev/null || tmux new-session -d -s regen -n run "LLM_OFFLINE=1 bash scripts/run_regen.sh; exec bash"
tmux ls
