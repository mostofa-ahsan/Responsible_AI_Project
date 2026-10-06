#!/usr/bin/env bash
cd ~/Responsible_AI_Project
echo "=== $(date '+%H:%M:%S') ===   tmux: $(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
echo "--- Stage status (local eval) ---"
[ -f results/local_eval/stage_status.tsv ] && tail -n 6 results/local_eval/stage_status.tsv | column -t -s $'\t' | cut -c1-110
echo "--- Latest events ---"
grep -a -E "\[orchestrator\]|FALLBACK|FAILED|Traceback|WARNING|committed|pushed" logs/localeval.log 2>/dev/null | tail -n 4 | cut -c1-140
echo "--- Live progress ---"
tail -c 4000 logs/localeval.log 2>/dev/null | tr -d '\000' | tr '\r' '\n' | grep -a -E "\[S[0-9]|it/s|s/it|units|pairs|/[0-9]+ " | tail -n 2 | cut -c1-140
echo "--- Seed replication ---"
L=$(ls -t logs/*seed*.log 2>/dev/null | head -1)
if [ -n "$L" ]; then
  tail -c 3000 "$L" | tr -d '\000' | tr '\r' '\n' | grep -a -E "step [0-9]+/|[0-9]+/[0-9]+ \[|done|seed" | tail -n 2 | cut -c1-140
else echo "not started yet"; fi
ls -d models_seeds/*/seed43/epoch1/adapter 2>/dev/null | sed 's/^/finished: /'
echo "--- GPU (util, mem, temp, power draw / limit) ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,power.limit --format=csv,noheader
[ -f logs/gpu_temp.log ] && awk -F', *' 'NR>1 && $2+0>m {m=$2+0} END {if (m) print "max temp so far: " m " C"}' logs/gpu_temp.log
echo "--- Disk / last commit ---"
df -h --output=avail / | tail -1 | sed 's/^ */free disk: /'
git log --oneline -1
