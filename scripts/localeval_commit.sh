#!/usr/bin/env bash
# Final commit + push of the local evaluation (called last by scripts/run_all_localeval.sh and by
# scripts/seeds_queue.sh at the end). If the push fails, the commits stay local and STATUS.md says so.
set +e
cd "$(dirname "$0")/.." || exit 1
STATUS=results/local_eval/STATUS.md
MSG=${1:-"Local evaluation: final results, paper pack and STATUS"}
bash scripts/git_local_commit.sh "$MSG"
if GIT_TERMINAL_PROMPT=0 timeout 180 git push origin HEAD 2>&1 | tail -2; [ "${PIPESTATUS[0]}" -eq 0 ]; then
  printf '\n- Git: pushed %s to origin at %s.\n' "$(git rev-parse --short HEAD)" "$(date '+%F %T')" >> "$STATUS"
else
  printf '\n- Git: committed %s locally; **push FAILED** at %s (see logs/localeval.log). Run `git push` manually.\n' \
    "$(git rev-parse --short HEAD)" "$(date '+%F %T')" >> "$STATUS"
fi
bash scripts/git_local_commit.sh "Local evaluation: STATUS.md push note"
GIT_TERMINAL_PROMPT=0 timeout 180 git push origin HEAD 2>&1 | tail -2
