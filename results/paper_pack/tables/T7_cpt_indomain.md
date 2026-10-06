**Table 7 (arm C). In-domain (unseen chunks) (n = 500)**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Llama 3.1 8B | base | 0.575 | 0.555 | 0.091 | 0.182 | 0.000 |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.510† | 0.418† | 0.102 | 0.322† | 0.000 |
| Llama 3.1 8B | QA-only ep1 | 0.525† | 0.372† | 0.087 | 0.358† | 0.002 |
| Llama 3.1 8B | QA-only ep2 | 0.496† | 0.371† | 0.093 | 0.338† | 0.000 |
| Llama 3.1 8B | QA-only ep3 | 0.514† | 0.371† | 0.077 | 0.334† | 0.000 |
| Llama 3.1 8B | mixed CPT ep1 | 0.539 | 0.374† | 0.069 | **0.380†‡** | 0.002 |
| Llama 3.1 8B | mixed CPT ep2 | 0.528 | 0.373† | 0.085 | 0.352†‡ | 0.000 |
| Qwen3-8B | base | 0.669 | 0.564 | 0.096 | 0.265 | 0.000 |
| Qwen3-8B | concise base (≤ 40 words) | 0.548† | 0.365† | 0.096 | 0.305† | 0.000 |
| Qwen3-8B | QA-only ep1 | 0.535† | 0.360† | 0.070 | 0.363† | 0.002 |
| Qwen3-8B | QA-only ep2 | 0.508† | 0.381† | 0.085 | 0.343† | 0.000 |
| Qwen3-8B | QA-only ep3 | 0.536† | 0.385† | 0.079 | 0.343† | 0.000 |
| Qwen3-8B | mixed CPT ep1 | 0.557† | 0.372† | 0.075 | 0.378†‡ | 0.004 |
| Gemma 4 E4B | base | **0.731** | **0.574** | 0.078 | 0.177 | 0.000 |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.595† | 0.367† | **0.067** | 0.287† | 0.000 |
| Gemma 4 E4B | QA-only ep1 | 0.553† | 0.371† | 0.069 | 0.369† | 0.006 |
| Gemma 4 E4B | QA-only ep2 | 0.530† | 0.364† | 0.070 | 0.351† | 0.000 |
| Gemma 4 E4B | QA-only ep3 | 0.532† | 0.369† | 0.076 | 0.340† | 0.000 |
| Gemma 4 E4B | mixed CPT ep1 | 0.547† | 0.365† | 0.073 | 0.377† | **0.008** |

QA-only = closed-book QLoRA r16 on QA pairs (models_epochs); mixed CPT = arm C, QLoRA on raw training-document text + QA. † differs from the same model's base, ‡ from QA-only at the same epoch (paired bootstrap, Holm-adjusted p < 0.05). Judge acc = (correct + 0.5 partial) / N; KF recall: MiniCheck-7B; Contra.: DeBERTa NLI; Exact repro. = ROUGE-L ≥ 0.8 vs the trained answer. '–' = not measured (see STATUS.md).
