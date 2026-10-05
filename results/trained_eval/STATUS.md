# Trained-question recall evaluation: STATUS

_Written 2026-10-05 11:31._

| Stage | Result | Wall time (min) | Note |
|---|---|---|---|
| T0_testset | done | 0 |  |
| T1_gen_vllm | done | 19 |  |
| T2_gen_nf4 | done | 32 |  |
| T3_purge_minicheck | done | 0 |  |
| T4_awq | done | 20 |  |
| T5_gguf | done | 0 |  |
| T6_judge_keyfacts | done | 7 |  |
| T7_grade | done | 68 |  |
| T8_checks | done | 17 |  |
| T9_stats | done | 0 |  |
| T10_pack | done | 0 |  |

## Decisions and fallbacks

- GPU power limit at start (2026-10-05 08:43:56): 260.00 W (260 W cap active)
- Best epoch per model (mean local-judge accuracy, earlier splits): {"qwen3-8b": "ep1", "gemma-4-e4b-it": "ep1", "llama-3.1-8b-instruct": "ep2"}
- Concise base: training system prompt + 'Answer in 40 words or fewer.'.
- T5: GGUF Q4_K_M variant skipped (llama.cpp is not installed; the only local GGUF route is Ollama, whose model store belongs to another project on this machine (not touched); a GGUF export needs the merged bf16 model on disk (~16 GB) plus the Q4_K_M file; 31.6 GB free; no CUDA toolkit to build llama.cpp's GPU backend (CPU-only inference would take hours))
- T6: gemma-4-e4b-it: 4-bit export redone with AWQ (RTN answers kept in answers/gemma-4-e4b-it__awq_ep1_rtn, not graded) (the first AWQ attempt failed because the Gemma 3 mapping regexes also matched the audio tower)
- T6: gemma-4-e4b-it: the 4-bit variant is RTN W4A16 (round-to-nearest, asym, group 128), not AWQ (llm-compressor 0.14 has no Gemma 4 AWQ mapping; Gemma 3 regexes also match the audio tower, and language-model-anchored mappings fail because Gemma 4's KV-shared layers have no v_proj. An MLP-only AWQ retry would need ~10 GB more disk while the judge is present (reserve 10 GB) and could not be graded after the judge is deleted)
- qwen3-8b__nf4_ep1: transformers + bitsandbytes NF4 (double quant) + PEFT LoRA, batch 16; tokens/s 92.8; weights VRAM 6.35 GB
- gemma-4-e4b-it__nf4_ep1: transformers + bitsandbytes NF4 (double quant) + PEFT LoRA, batch 16; tokens/s 64.9; weights VRAM 9.52 GB
- llama-3.1-8b-instruct__nf4_ep2: transformers + bitsandbytes NF4 (double quant) + PEFT LoRA, batch 16; tokens/s 108.9; weights VRAM 5.97 GB
- qwen3-8b__awq_ep1: vLLM, compressed-tensors AWQ W4A16 asym (group 128); tokens/s 1582.0; weights VRAM 5.71 GB
- gemma-4-e4b-it__awq_ep1: vLLM, compressed-tensors RTN W4A16 asym (AWQ failed: ValueError); tokens/s 2307.8; weights VRAM 9.72 GB
- llama-3.1-8b-instruct__awq_ep2: vLLM, compressed-tensors AWQ W4A16 asym (group 128); tokens/s 1875.0; weights VRAM 5.39 GB
- Power-limit history (logs/gpu_temp.log): 420.00 W from 2026-10-04 21:08:04; 260.00 W from 2026-10-04 21:26:47.
- GPU (logs/gpu_temp.log, 1711 readings since 2026-10-04 21:08:04): max 76 °C at 2026-10-04 21:26:19, max power draw 417 W at 2026-10-04 21:24:24; thermal pauses: 0.

Outputs: `results/trained_eval/` (SUMMARY via metrics_by_system.csv, significance.csv, robustness_gap.csv, deployment.csv), paper pack Table 7/7b/7c/8 and Figures 9/10, `results/seed_replication/SUMMARY.md` (seed-43 judge accuracy).
