**Table 7 (arm C). Held-out documents (n = 500)**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Llama 3.1 8B | base | 0.534 | 0.493 | 0.100 | 0.179 | 0.000 |
| Llama 3.1 8B | QA-only ep1 | 0.460† | 0.325† | 0.093 | 0.334† | 0.002 |
| Llama 3.1 8B | QA-only ep2 | 0.465† | 0.311† | 0.102 | 0.310† | 0.002 |
| Llama 3.1 8B | QA-only ep3 | 0.462† | 0.308† | 0.090 | 0.311† | 0.002 |
| Qwen3-8B | base | 0.642 | 0.518 | 0.088 | 0.259 | 0.000 |
| Qwen3-8B | QA-only ep1 | 0.506† | 0.345† | 0.101 | 0.340† | **0.004** |
| Qwen3-8B | QA-only ep2 | 0.477† | 0.323† | 0.085 | 0.319† | 0.002 |
| Qwen3-8B | QA-only ep3 | 0.476† | 0.337† | 0.091 | 0.315† | 0.002 |
| Gemma 4 E4B | base | **0.679** | **0.541** | **0.065** | 0.177 | 0.000 |
| Gemma 4 E4B | QA-only ep1 | 0.500† | 0.322† | 0.096 | **0.347†** | 0.002 |
| Gemma 4 E4B | QA-only ep2 | 0.498† | 0.327† | 0.093 | 0.332† | 0.002 |
| Gemma 4 E4B | QA-only ep3 | 0.484† | 0.335† | 0.085 | 0.326† | 0.002 |

QA-only = closed-book QLoRA r16 on QA pairs (models_epochs); mixed CPT = arm C, QLoRA on raw training-document text + QA. † differs from the same model's base, ‡ from QA-only at the same epoch (paired bootstrap, Holm-adjusted p < 0.05). Judge acc = (correct + 0.5 partial) / N; KF recall: MiniCheck-7B; Contra.: DeBERTa NLI; Exact repro. = ROUGE-L ≥ 0.8 vs the trained answer. '–' = not measured (see STATUS.md).
