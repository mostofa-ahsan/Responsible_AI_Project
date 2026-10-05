# Local evaluation: STATUS

_Written 2026-10-04 21:38 by `src/local_pack.py status`._

| Stage | Result | Wall time (min) | Note |
|---|---|---|---|
| S1_calibration | done | 40 | gate recomputed from cached grades under the refined rule (judge A); old process stopped before Phi-4 |
| S2_grading | done | 29 |  |
| S3_keyfacts | done | 4 |  |
| S4_claims | done | 45 |  |
| S5a_purge_judge | done | 0 |  |
| S5b_minicheck7b | done | 12 |  |
| S5c_flant5 | done | 1 |  |
| S5d_nli | done | 10 |  |
| S6_stats | done | 0 |  |
| S8_rag_optional | done | 0 |  |
| S7_pack | done | 0 |  |

## Decisions and fallbacks

- Judge: **mistral-small-3.2-24b-awq** prompt v1; gate passed; candidates {"A": {"max_binary_kappa": 0.6064, "passes_gate": true}}; judge B: not needed: judge A passes the refined gate
- S3_keyfacts_info: {"decomposed": 1279, "total": 1279}
- S4_claims_info: {"decomposed": 16199, "total": 19185}
- S2_grade: {"graded": 19176, "total": 19185}
- S5b: Flan-T5-Large is the primary support checker (MiniCheck-7B covered only 0/67500 pairs)
- S5b: reverted: MiniCheck-7B is the primary support checker again (rerun after the post-crash relaunch with the vLLM InternLM2 fix (src/vllm_shims/sitecustomize.py))
- S8 (RAG baseline): skipped; the judge weights were deleted after S4 by design, so RAG answers could not be graded with the same calibrated judge in this run; re-downloading it (15 GB) plus the retrieval models does not fit the disk reserve next to MiniCheck-7B. Future work: closed-book vs RAG baseline (BM25 + dense + Qwen3-Reranker, recall@5 92%) graded by the same local judge and checkers
- **WARNING (power limit):** 2026-10-04 21:16:15 power limit 420 W enforced (reported by both WSL and Windows nvidia-smi.exe; draw up to ~416 W under load). The 260 W cap set in Windows is not active (reset by the reboot?). Setting it from here failed: 'Insufficient Permissions' (needs an elevated Windows prompt: nvidia-smi -pl 260).
- GPU (logs/gpu_temp.log, 64 readings since 2026-10-04 21:08:04): max 76 °C at 2026-10-04 21:26:19, max power draw 417 W at 2026-10-04 21:24:24; thermal pauses: 0.
- The PC hard-crashed at about 12:13 (power cut under GPU load). Relaunched at 21:14 with scripts/resume_after_crash.sh logic: finished stages kept, S5b rerun with the vLLM InternLM2 fix, S5c finished its remaining pairs, S5d rerun with incremental writes.
- Seed replication (seed 43, epoch 1; tmux `seeds`): finished steps none yet; see `results/seed_replication/SUMMARY.md`.

## Where to look

- `results/paper_pack/RESULTS.md` (write-up), `results/paper_pack/README.md` (file index)
- `results/local_eval/SUMMARY.md`, `judge_calibration.md`, `keyfact_validation.md`, `metrics_by_system.csv`
- Human review: `results/local_eval/keyfact_spotcheck.xlsx`, `results/paper_pack/human_eval_sheet.xlsx`
