> **WARNING: GPU power limit above 270 W** during this run (39 readings, first 2026-10-04 21:08:04). The 260 W cap resets on reboot and cannot be set from WSL.

# Arm C (mixed continued pretraining): STATUS

_Written 2026-10-05 22:18._

| Stage | Result | Wall time (min) | Note |
|---|---|---|---|
| C1_gen_concise | not run | – | |
| C1b_checks_concise | not run | – | |
| C2_build_mix | not run | – | |
| C3_llama-3.1-8b-instruct | not run | – | |
| C3_qwen3-8b | not run | – | |
| C3_gemma-4-e4b-it | not run | – | |
| C4_judge | not run | – | |
| C5_checks_final | not run | – | |
| C6_report | not run | – | |

## Decisions, fallbacks and notes

- Stage 1 reuse: data/splits/test_trained_exact.jsonl and its key facts were built earlier with the same rule (500 train items, all 279 seen-facts sources, stratified q_type × dimension, ≤ 8 per doc for added items, seed 42); answers of base / QA-only ep1-3 / 1-epoch adapters on it were generated earlier with identical settings and are reused. The concise base was regenerated with the exact prompt 'Answer in at most 40 words.' on all 4 test types.
- Seed-43 answers were already graded by the same judge (results/seed_replication/SUMMARY.md); not re-graded.
- Throughput: with the GPU pinned at the 260 W cap, arm C trains at ~9 s per optimizer step (~2.6 h per epoch), so 3 epochs (~8 h) exceed the 6 h per-model box: an epoch is started only if it can finish inside the box; the cosine schedule still spans 3 epochs, so a final epoch-2 adapter is a mid-schedule checkpoint.
- GPU during this run: max 76 °C (2026-10-04 21:26:19), max power draw 417 W (2026-10-04 21:24:24), power limit 260–420 W; thermal pauses: 0.
- Disk free at the end: 17.5 GB.
- Git: branch exp/cpt-mixed; 0 failed push attempt(s) recorded (see logs/cpt.log).

Recovery after a crash: `bash scripts/resume_cpt_after_crash.sh`.
