**Table 8. Deployment variants of each model's best epoch**

| Model | Deployment | Disk (GB) | Weights VRAM (GB) | Tokens/s | Judge acc trained_exact | Judge acc seen_facts | Judge acc heldout_docs | Accuracy retained vs bf16 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | bf16 base + LoRA (vLLM) | 16.5 | 15.4 | 1261 | 0.593 | 0.573 | 0.506 | 100.0% |
| Qwen3-8B | NF4 base + LoRA (transformers) | 16.5 | 6.3 | 93 | 0.590 | 0.597 | 0.475 | 99.2% |
| Qwen3-8B | merged 4-bit (vLLM) | 6.1 | 5.7 | 1582 | 0.572 | 0.556 | 0.485 | 96.4% |
| Gemma 4 E4B | bf16 base + LoRA (vLLM) | 16.1 | 15.3 | 1894 | 0.563 | 0.595 | 0.500 | 100.0% |
| Gemma 4 E4B | NF4 base + LoRA (transformers) | 16.1 | 9.5 | 65 | 0.576 | 0.573 | 0.499 | 99.5% |
| Gemma 4 E4B | merged 4-bit (vLLM) | 10.0 | 9.7 | 2308 | 0.528 | 0.550 | 0.485 | 94.4% |
| Llama 3.1 8B | bf16 base + LoRA (vLLM) | 16.1 | 15.1 | 1470 | 0.853 | 0.674 | 0.465 | 100.0% |
| Llama 3.1 8B | NF4 base + LoRA (transformers) | 16.1 | 6.0 | 109 | 0.962 | 0.722 | 0.451 | 105.6% |
| Llama 3.1 8B | merged 4-bit (vLLM) | 5.8 | 5.4 | 1875 | 0.687 | 0.596 | 0.441 | 87.9% |

Best epoch per model by mean local-judge accuracy over the earlier test splits (Qwen3-8B ep1, Gemma 4 E4B ep1, Llama 3.1 8B ep2). Disk: base checkpoint + adapter (NF4 quantizes the bf16 checkpoint at load time) or the exported 4-bit model. Weights VRAM: vLLM 'model loading' memory, or torch peak allocation after loading for NF4. Tokens/s: batched greedy generation throughput on this workload (vLLM batches the whole split; transformers batch 16), so it compares serving stacks rather than kernels. Retained = mean over the 3 splits of variant / bf16 judge accuracy.
