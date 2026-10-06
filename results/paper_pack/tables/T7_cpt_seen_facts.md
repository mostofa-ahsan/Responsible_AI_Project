**Table 7 (arm C). Paraphrased training questions (n = 279)**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Llama 3.1 8B | base | 0.522 | 0.498 | 0.083 | 0.154 | 0.000 |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.530 | 0.356† | 0.115 | 0.265† | 0.000 |
| Llama 3.1 8B | QA-only ep1 | 0.615† | 0.387† | 0.078 | 0.363† | 0.025 |
| Llama 3.1 8B | QA-only ep2 | **0.674†** | 0.487 | 0.065 | 0.463† | 0.125† |
| Llama 3.1 8B | QA-only ep3 | 0.654† | 0.485 | **0.058** | **0.473†** | **0.143†** |
| Llama 3.1 8B | mixed CPT ep1 | 0.572 | 0.372† | 0.081 | 0.324†‡ | 0.007 |
| Llama 3.1 8B | mixed CPT ep2 | 0.620† | 0.419†‡ | 0.081 | 0.398†‡ | 0.075† |
| Qwen3-8B | base | 0.649 | 0.523 | 0.108 | 0.213 | 0.000 |
| Qwen3-8B | concise base (≤ 40 words) | 0.582† | 0.316† | 0.105 | 0.263† | 0.000 |
| Qwen3-8B | QA-only ep1 | 0.573† | 0.338† | 0.086 | 0.336† | 0.011 |
| Qwen3-8B | QA-only ep2 | 0.590 | 0.426† | 0.083 | 0.366† | 0.025 |
| Qwen3-8B | QA-only ep3 | 0.584 | 0.407† | 0.078 | 0.357† | 0.025 |
| Qwen3-8B | mixed CPT ep1 | 0.575† | 0.333† | 0.082 | 0.311†‡ | 0.004 |
| Gemma 4 E4B | base | 0.668 | **0.530** | 0.063 | 0.151 | 0.000 |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.595† | 0.303† | 0.077 | 0.248† | 0.000 |
| Gemma 4 E4B | QA-only ep1 | 0.595† | 0.359† | 0.085 | 0.340† | 0.014 |
| Gemma 4 E4B | QA-only ep2 | 0.599 | 0.386† | 0.090 | 0.359† | 0.018 |
| Gemma 4 E4B | QA-only ep3 | 0.629 | 0.402† | 0.086 | 0.368† | 0.018 |
| Gemma 4 E4B | mixed CPT ep1 | 0.554† | 0.317† | 0.100 | 0.305†‡ | 0.004 |

QA-only = closed-book QLoRA r16 on QA pairs (models_epochs); mixed CPT = arm C, QLoRA on raw training-document text + QA. † differs from the same model's base, ‡ from QA-only at the same epoch (paired bootstrap, Holm-adjusted p < 0.05). Judge acc = (correct + 0.5 partial) / N; KF recall: MiniCheck-7B; Contra.: DeBERTa NLI; Exact repro. = ROUGE-L ≥ 0.8 vs the trained answer. '–' = not measured (see STATUS.md).
