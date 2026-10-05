#!/usr/bin/env bash
# One command after a crash or reboot: relaunches everything that is not finished. Safe to run any time
# (each piece skips finished work via markers and caches; running sessions are left alone).
#   bash scripts/resume_after_crash.sh
cd "$(dirname "$0")/.." || exit 1
for f in logs/localeval.log logs/seeds.log logs/localeval_worker.log; do     # a power cut leaves NUL bytes
  [ -f "$f" ] && tr -d '\000' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
rm -f results/local_eval/judge_cache/*.in.jsonl results/local_eval/judge_cache/*.tmp
echo "===== $(date '+%F %T') resume_after_crash.sh" >> logs/localeval.log
tmux has-session -t watchdog 2>/dev/null || tmux new-session -d -s watchdog -n gpu "bash scripts/gpu_watchdog.sh"
if [ ! -f results/local_eval/.done/PART1_DONE ]; then
  # a stage killed mid-run left no marker and reruns; the paper pack and stats are always rebuilt
  rm -f results/local_eval/.done/S6_stats
  tmux has-session -t localeval 2>/dev/null || \
    tmux new-session -d -s localeval -n part1 "LLM_OFFLINE=1 bash scripts/run_all_localeval.sh; exec bash"
fi
if ! grep -q "seeds_queue.sh finished" logs/seeds.log 2>/dev/null; then
  tmux has-session -t seeds 2>/dev/null || \
    tmux new-session -d -s seeds -n queue "LLM_OFFLINE=1 bash scripts/seeds_queue.sh; exec bash"
fi
tmux ls
echo "Watch: tail -f logs/localeval.log | tail -f logs/seeds.log | tail -f logs/gpu_temp.log"
