**Table 8. Deployment variants of each model's best epoch**

| Model | Deployment | Disk (GB) | Weights VRAM (GB) | Tokens/s | Judge acc trained_exact | Judge acc seen_facts | Judge acc heldout_docs | Accuracy retained vs bf16 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | bf16 base + LoRA (vLLM) | 16.5 | 15.4 | 1261 | – | 0.573 | 0.506 | – |
| Qwen3-8B | NF4 base + LoRA (transformers) | 16.5 | – | – | – | – | – | – |
| Qwen3-8B | merged 4-bit (vLLM) | – | – | – | – | – | – | – |
| Gemma 4 E4B | bf16 base + LoRA (vLLM) | 16.1 | – | – | – | 0.595 | 0.500 | 100.0% |
| Gemma 4 E4B | NF4 base + LoRA (transformers) | 16.1 | – | – | – | – | – | – |
| Gemma 4 E4B | merged 4-bit (vLLM) | – | – | – | – | – | – | – |
| Llama 3.1 8B | bf16 base + LoRA (vLLM) | 16.1 | – | – | – | 0.674 | 0.465 | 100.0% |
| Llama 3.1 8B | NF4 base + LoRA (transformers) | 16.1 | – | – | – | – | – | – |
| Llama 3.1 8B | merged 4-bit (vLLM) | – | – | – | – | – | – | – |

Best epoch per model by mean local-judge accuracy over the earlier test splits (Qwen3-8B ep1, Gemma 4 E4B ep1, Llama 3.1 8B ep2). Disk: base checkpoint + adapter (NF4 quantizes the bf16 checkpoint at load time) or the exported 4-bit model. Weights VRAM: vLLM 'model loading' memory, or torch peak allocation after loading for NF4. Tokens/s: batched greedy generation throughput on this workload (vLLM batches the whole split; transformers batch 16), so it compares serving stacks rather than kernels. Retained = mean over the 3 splits of variant / bf16 judge accuracy.
