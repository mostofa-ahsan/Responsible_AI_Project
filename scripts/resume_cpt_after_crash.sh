#!/usr/bin/env bash
# After a crash / reboot: relaunch whatever is unfinished in the arm-C (mixed continued pretraining) experiment.
# Safe to run any time: finished stages are skipped (results/cpt/.done/), training resumes from the latest checkpoint
# (every 100 steps), generation and checks only do missing items. Check the GPU power cap in Windows first.
#   bash scripts/resume_cpt_after_crash.sh
cd "$(dirname "$0")/.." || exit 1
[ "$(git rev-parse --abbrev-ref HEAD)" = "exp/cpt-mixed" ] || { echo "switch to branch exp/cpt-mixed first"; exit 1; }
for f in logs/cpt.log logs/train_cpt.log logs/trained_eval_worker.log logs/localeval.log; do   # power cuts leave NULs
  [ -f "$f" ] && tr -d '\000' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
rm -f results/cpt/raw/*.in.jsonl
tmux has-session -t watchdog 2>/dev/null || tmux new-session -d -s watchdog -n watchdog "bash scripts/gpu_watchdog.sh"
if [ -f results/cpt/.done/ALL_DONE ]; then echo "arm C experiment already finished"; exit 0; fi
tmux has-session -t cpt 2>/dev/null || tmux new-session -d -s cpt -n run "LLM_OFFLINE=1 bash scripts/run_cpt.sh; exec bash"
tmux ls
echo "Watch: tail -f logs/cpt.log   (GPU: tail -f logs/gpu_temp.log)"
