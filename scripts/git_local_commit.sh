#!/usr/bin/env bash
# Local commit of the local-evaluation / seed-replication code and results (no push; never weights or caches).
#   bash scripts/git_local_commit.sh "message"
set +e
cd "$(dirname "$0")/.." || exit 1
MSG=${1:-"Local evaluation: checkpoint commit"}
if ! grep -q "# local evaluation (added by localeval_commit.sh)" .gitignore; then
  cat >> .gitignore <<'EOF'

# local evaluation (added by localeval_commit.sh): results and figures in, caches out
!results/local_eval/
!results/local_eval/per_item/
!results/local_eval/per_item/*.jsonl
results/local_eval/judge_cache/
results/local_eval/.done/
results/local_eval/*.tmp
!results/paper_pack/
!results/paper_pack/figures/
!results/paper_pack/tables/
!data/eval_keyfacts/
!data/eval_keyfacts/*.jsonl
EOF
fi
if ! grep -q "trained_eval (added" .gitignore; then
  printf '\n# trained_eval (added by git_local_commit.sh): metrics in, answers/caches/quantized models out\n!results/trained_eval/\n!results/trained_eval/per_item/\n!results/trained_eval/per_item/*.jsonl\nresults/trained_eval/.done/\nresults/trained_eval/answers/\nresults/trained_eval/raw/\nresults/trained_eval/cache/\nmodels_quant/\n' >> .gitignore
fi
if ! grep -q "cpt (added" .gitignore; then
  printf '\n# cpt (added by git_local_commit.sh): summaries + per-item results in; answers, caches, corpus, adapters out\n!results/cpt/\n!results/cpt/per_item/\n!results/cpt/per_item/*.jsonl\nresults/cpt/.done/\nresults/cpt/answers/\nresults/cpt/raw/\nresults/cpt/cache/\nmodels_cpt*/\ndata/cpt*/\n' >> .gitignore
fi
if ! grep -q "^models_seeds" .gitignore; then
  printf '\n# seed replication: adapters stay local, summaries are committed\nmodels_seeds*/\nresults/seed_replication/.done/\nresults/.gpu_pause\n' >> .gitignore
fi
# GitHub rejects files > 100 MB: keep any such file out
for f in $(find results/local_eval results/paper_pack results/seed_replication results/trained_eval results/cpt data/eval_keyfacts -type f -size +45M 2>/dev/null); do
  grep -qxF "$f" .git/info/exclude || echo "$f" >> .git/info/exclude
done
paths=(.gitignore CLAUDE.md README.md src/local_*.py src/seed_eval.py src/train_epochs.py src/vllm_json_worker.py
       src/vllm_shims scripts/run_all_localeval.sh scripts/localeval_commit.sh scripts/git_local_commit.sh
       scripts/gpu_watchdog.sh scripts/resume_after_crash.sh scripts/seeds_queue.sh
       results/local_eval results/paper_pack results/seed_replication data/eval_keyfacts results/run_epochs/PROGRESS.md
       src/trained_eval.py src/trained_quant.py src/vllm_multi_generate.py scripts/run_trained_eval.sh
       results/trained_eval data/splits/test_trained_exact.jsonl
       src/build_cpt_mix.py src/train_cpt.py src/cpt.py src/cpt_report.py scripts/run_cpt.sh
       scripts/resume_cpt_after_crash.sh results/cpt)
existing=(); for p in "${paths[@]}"; do [ -e "$p" ] && existing+=("$p"); done   # git add aborts on a missing path
git add "${existing[@]}" 2>&1 | grep -v "^$" | head -3
bad=$(git diff --cached --name-only | grep -E '\.(safetensors|bin|pt|gguf)$|judge_cache/|/\.done/|\.gpu_pause')
if [ -n "$bad" ]; then echo "$bad" | xargs git reset -q HEAD --; fi
if git diff --cached --quiet; then echo "git: nothing new to commit"; exit 0; fi
git commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "git: local commit $(git rev-parse --short HEAD): $MSG"
