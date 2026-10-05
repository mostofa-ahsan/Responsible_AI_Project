# Trained-question recall evaluation: STATUS

_Written 2026-10-05 08:53._

| Stage | Result | Wall time (min) | Note |
|---|---|---|---|
| T0_testset | done | 0 |  |
| T1_gen_vllm | not run | – | |
| T2_gen_nf4 | not run | – | |
| T3_purge_minicheck | not run | – | |
| T4_awq | not run | – | |
| T5_gguf | not run | – | |
| T6_judge_keyfacts | not run | – | |
| T7_grade | not run | – | |
| T8_checks | not run | – | |
| T9_stats | not run | – | |
| T10_pack | not run | – | |

## Decisions and fallbacks

- GPU power limit at start (2026-10-05 08:43:56): 260.00 W (260 W cap active)
- Best epoch per model (mean local-judge accuracy, earlier splits): {"qwen3-8b": "ep1", "gemma-4-e4b-it": "ep1", "llama-3.1-8b-instruct": "ep2"}
- Concise base: training system prompt + 'Answer in 40 words or fewer.'.
- Power-limit history (logs/gpu_temp.log): 420.00 W from 2026-10-04 21:08:04; 260.00 W from 2026-10-04 21:26:47.
- GPU (logs/gpu_temp.log, 1396 readings since 2026-10-04 21:08:04): max 76 °C at 2026-10-04 21:26:19, max power draw 417 W at 2026-10-04 21:24:24; thermal pauses: 0.

Outputs: `results/trained_eval/` (SUMMARY via metrics_by_system.csv, significance.csv, robustness_gap.csv, deployment.csv), paper pack Table 7/7b/7c/8 and Figures 9/10, `results/seed_replication/SUMMARY.md` (seed-43 judge accuracy).
