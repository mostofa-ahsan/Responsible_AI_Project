**Table 7 (arm C). Exact training questions (n = 500)**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Llama 3.1 8B | base | 0.552 | 0.517 | 0.100 | 0.179 | 0.000 |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.478† | 0.369† | 0.117 | 0.324† | 0.000 |
| Llama 3.1 8B | QA-only ep1 | 0.607† | 0.430† | 0.058† | 0.481† | 0.080† |
| Llama 3.1 8B | QA-only ep2 | 0.853† | 0.726† | 0.029† | 0.811† | 0.658† |
| Llama 3.1 8B | QA-only ep3 | **0.896†** | **0.751†** | **0.019†** | **0.856†** | **0.742†** |
| Llama 3.1 8B | mixed CPT ep1 | 0.564 | 0.406† | 0.066 | 0.426†‡ | 0.026‡ |
| Llama 3.1 8B | mixed CPT ep2 | 0.735†‡ | 0.583†‡ | 0.033† | 0.619†‡ | 0.308†‡ |
| Qwen3-8B | base | 0.657 | 0.559 | 0.093 | 0.266 | 0.000 |
| Qwen3-8B | concise base (≤ 40 words) | 0.537† | 0.337† | 0.096 | 0.320† | 0.004 |
| Qwen3-8B | QA-only ep1 | 0.593† | 0.399† | 0.069 | 0.445† | 0.054† |
| Qwen3-8B | QA-only ep2 | 0.667 | 0.511† | 0.048† | 0.537† | 0.184† |
| Qwen3-8B | QA-only ep3 | 0.670 | 0.529 | 0.055 | 0.561† | 0.214† |
| Qwen3-8B | mixed CPT ep1 | 0.569† | 0.363† | 0.071 | 0.408†‡ | 0.022‡ |
| Gemma 4 E4B | base | 0.702 | 0.562 | 0.073 | 0.173 | 0.000 |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.546† | 0.316† | 0.082 | 0.289† | 0.002 |
| Gemma 4 E4B | QA-only ep1 | 0.563† | 0.365† | 0.073 | 0.422† | 0.026 |
| Gemma 4 E4B | QA-only ep2 | 0.614† | 0.422† | 0.068 | 0.467† | 0.066† |
| Gemma 4 E4B | QA-only ep3 | 0.638† | 0.432† | 0.061 | 0.486† | 0.090† |
| Gemma 4 E4B | mixed CPT ep1 | 0.534† | 0.333† | 0.076 | 0.394†‡ | 0.010 |

QA-only = closed-book QLoRA r16 on QA pairs (models_epochs); mixed CPT = arm C, QLoRA on raw training-document text + QA. † differs from the same model's base, ‡ from QA-only at the same epoch (paired bootstrap, Holm-adjusted p < 0.05). Judge acc = (correct + 0.5 partial) / N; KF recall: MiniCheck-7B; Contra.: DeBERTa NLI; Exact repro. = ROUGE-L ≥ 0.8 vs the trained answer. '–' = not measured (see STATUS.md).
