# Arm C (mixed continued pretraining): STATUS

_Written 2026-10-06 15:04._

| Stage | Result | Wall time (min) | Note |
|---|---|---|---|
| C1_gen_concise | done | 6 |  |
| C1b_checks_concise | done | 5 |  |
| C2_build_mix | done | 0 | built and audited before launch |
| C3_llama-3.1-8b-instruct | done | 324 |  |
| C3_qwen3-8b | done | 299 |  |
| C3_gemma-4-e4b-it | done | 315 |  |
| C4_judge | done | 36 |  |
| C5_checks_final | done | 4 |  |
| C6_report | done | 0 |  |

## Decisions, fallbacks and notes

- Stage 1 reuse: data/splits/test_trained_exact.jsonl and its key facts were built earlier with the same rule (500 train items, all 279 seen-facts sources, stratified q_type × dimension, ≤ 8 per doc for added items, seed 42); answers of base / QA-only ep1-3 / 1-epoch adapters on it were generated earlier with identical settings and are reused. The concise base was regenerated with the exact prompt 'Answer in at most 40 words.' on all 4 test types.
- Seed-43 answers were already graded by the same judge (results/seed_replication/SUMMARY.md); not re-graded.
- Throughput: with the GPU pinned at the 260 W cap, arm C trains at ~9 s per optimizer step (~2.6 h per epoch), so 3 epochs (~8 h) exceed the 6 h per-model box: an epoch is started only if it can finish inside the box; the cosine schedule still spans 3 epochs, so a final epoch-2 adapter is a mid-schedule checkpoint.
- C3_llama-3.1-8b-instruct (2026-10-06 03:51:15): epoch 3 not started: 2364s left in the 6 h box, epoch needs ~10024s (finished epochs kept)
- C3_qwen3-8b (2026-10-06 08:52:06): epoch 2 not started: 3805s left in the 6 h box, epoch needs ~18391s (finished epochs kept)
- C3_qwen3-8b (2026-10-06 08:52:58): epoch 1 ran at ~16 s/step (282 min) vs ~8.4 s/step for Llama, steady through the epoch; peak VRAM 22.6 of 24 GB at micro-batch 4, so the WSL driver likely spilled into shared system memory. Only epoch 1 fits the 6 h box. A rerun should use micro-batch 2 (same effective batch 16)
- C3_gemma-4-e4b-it (2026-10-06 09:32:00): at r=128, micro-batch 1 Gemma filled 24.3 of 24.6 GB VRAM and spilled into shared memory (33 s/step, 177 W): one epoch ~9.8 h > the 6 h box. Treated as the OOM condition -> spec fallback LoRA r=64, alpha=128 (REPORT IN PAPER); also PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator only). Box start unchanged (08:54)
- C3_gemma-4-e4b-it (2026-10-06 09:32:03): deleted MiniCheck-7B early (free disk < 15 GB); key-fact recall for later models moves to Stage 4
- C3_gemma-4-e4b-it (2026-10-06 09:52:14): restarted fresh (r=64): the Trainer does not compute the token count for Gemma 4 (model_accepts_loss_kwargs=False), so its loss was a per-example mean at micro-batch 1 instead of the token-level mean used for Llama/Qwen; fixed by forcing the token count (train_cpt.py) so all models share the same loss weighting. Box start unchanged
- C3_gemma-4-e4b-it (2026-10-06 14:09:31): epoch 2 not started: 2723s left in the 6 h box, epoch needs ~15864s (finished epochs kept)
- C4_judge (2026-10-06 14:19:16): first C4 attempt failed at vLLM engine start (allocation error) because PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set globally at the Gemma restart) was inherited by the vLLM worker; C5 then deleted the judge before the pipeline was stopped. Fix: the setting now applies to training only; the judge had to be downloaded a SECOND time (2.2 min) and passed a 6-prompt smoke test (6/6 parsed) before grading
- C4_judge (2026-10-06 14:22:30): second C4 attempt failed its own disk pre-check (it required 26 GB free although the judge was already cached) and C5 deleted the judge again. Fixed: the headroom check applies only when the judge is not cached, and C5 deletes the judge only after C4 succeeded. Judge downloaded a THIRD time (2.2 min). The rule 'each big model downloaded at most once' was therefore not met for the judge
- Llama 3.1 8B: rank 128, micro-batch 4, epochs 1 (149.6 min, val QA 1.7109, held-out ppl 9.974), 2 (148.4 min, val QA 1.9057, held-out ppl 10.553); base held-out ppl 10.834
- Qwen3-8B: rank 128, micro-batch 4, epochs 1 (282.4 min, val QA 1.6974, held-out ppl 8.911); base held-out ppl 10.365
- Gemma 4 E4B: rank 64, micro-batch 1, epochs 1 (242.3 min, val QA 1.7509, held-out ppl 10.247); base held-out ppl 46.472
- GPU during this run: max 58 °C (2026-10-06 14:58:13), max power draw 262 W (2026-10-06 09:25:11), power limit 260–260 W; thermal pauses: 0.
- Disk free at the end: 11.4 GB.
- Git: branch exp/cpt-mixed; 0 failed push attempt(s) recorded (see logs/cpt.log).

Recovery after a crash: `bash scripts/resume_cpt_after_crash.sh`.
