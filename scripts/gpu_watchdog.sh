#!/usr/bin/env bash
# GPU watchdog (own tmux window). Every 30 s appends "time, temp, power draw, power limit, util, mem" to
# logs/gpu_temp.log and fsyncs it, so the last readings survive a power cut.
# - power limit > 270 W: WARNING line in the log and results/local_eval/power_warning.txt (STATUS.md shows it;
#   the Windows-side cap resets on reboot and is not always visible from WSL)
# - temp >= 84 C for 2+ min: creates results/.gpu_pause; GPU jobs finish their current chunk and then sleep in
#   5-min steps while it exists. Removed once temp < 75 C.
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gpu_temp.log
PAUSE=results/.gpu_pause
WARN=results/local_eval/power_warning.txt
mkdir -p results/local_eval logs
append() { printf '%s\n' "$1" | dd of="$LOG" oflag=append conv=notrunc,fsync status=none; }
[ -s "$LOG" ] || append "time,temp_c,power_draw_w,power_limit_w,util_pct,mem_used_mib"
hot=0
while true; do
  q=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used \
      --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -z "$q" ]; then append "$(date '+%F %T'),nvidia-smi failed"; sleep 30; continue; fi
  IFS=, read -r t pd pl u m <<< "$q"
  append "$(date '+%F %T'),$t,$pd,$pl,$u,$m"
  if [ "${pl%.*}" -gt 270 ] 2>/dev/null && [ ! -f "$WARN" ]; then
    append "$(date '+%F %T'),WARNING power limit ${pl} W > 270 W (cap not active as seen from WSL)"
    echo "$(date '+%F %T') power limit reported by nvidia-smi: ${pl} W (> 270 W); the Windows cap is not visible from WSL or was reset" > "$WARN"
  fi
  if [ "${t:-0}" -ge 84 ]; then hot=$((hot + 1)); else hot=0; fi
  if [ $hot -ge 4 ] && [ ! -f "$PAUSE" ]; then
    touch "$PAUSE"; append "$(date '+%F %T'),PAUSE temp ${t} C >= 84 for 2 min: new GPU work paused"
    echo "$(date +%H:%M:%S) [watchdog] GPU ${t} C for 2 min: pausing new GPU work" >> logs/localeval.log
  fi
  if [ -f "$PAUSE" ] && [ "${t:-99}" -lt 75 ]; then
    rm -f "$PAUSE"; append "$(date '+%F %T'),RESUME temp ${t} C < 75"
    echo "$(date +%H:%M:%S) [watchdog] GPU ${t} C: resuming" >> logs/localeval.log
  fi
  sleep 30
done
