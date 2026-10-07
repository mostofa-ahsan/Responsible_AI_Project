**Table 7. Reworded training questions: adapters served on their training base**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.649 | 0.523 | 0.108 | 0.213 | 0.000 |
| Qwen3-8B | concise base (≤ 40 words) | 0.582† | 0.316† | 0.105 | 0.263† | 0.000 |
| Qwen3-8B | QA-only ep1 | 0.599 | 0.338† | 0.078 | 0.351† | 0.014 |
| Qwen3-8B | QA-only ep2 | 0.627 | 0.404† | 0.091 | 0.387† | 0.061† |
| Qwen3-8B | QA-only ep3 | 0.619 | 0.402† | 0.077 | 0.374† | 0.050 |
| Qwen3-8B | mixed CPT ep1 | – | – | – | 0.311†‡ | 0.011 |
| Qwen3-8B | base on NF4 weights (control) | – | – | – | 0.216 | 0.000 |
| Gemma 4 E4B | base | 0.668 | 0.530 | 0.063 | 0.151 | 0.000 |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.595† | 0.303† | 0.077 | 0.248† | 0.000 |
| Gemma 4 E4B | QA-only ep1 | 0.593† | 0.366† | 0.087 | 0.343† | 0.018 |
| Gemma 4 E4B | QA-only ep2 | 0.622 | 0.414† | 0.084 | 0.398† | 0.075† |
| Gemma 4 E4B | QA-only ep3 | 0.613 | 0.435† | 0.072 | 0.393† | 0.075† |
| Gemma 4 E4B | mixed CPT ep1 | – | – | – | 0.297†‡ | 0.004 |
| Gemma 4 E4B | base on NF4 weights (control) | – | – | – | 0.160† | 0.000 |
| Llama 3.1 8B | base | 0.522 | 0.498 | 0.083 | 0.154 | 0.000 |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.530 | 0.356† | 0.115 | 0.265† | 0.000 |
| Llama 3.1 8B | QA-only ep1 | 0.609† | 0.407† | 0.073 | 0.372† | 0.043 |
| Llama 3.1 8B | QA-only ep2 | **0.719†** | 0.544 | 0.046 | 0.544†§ | 0.265†§ |
| Llama 3.1 8B | QA-only ep3 | **0.719†§** | **0.555§** | **0.045** | **0.548†§** | **0.269†§** |
| Llama 3.1 8B | mixed CPT ep1 | – | – | – | 0.326†‡ | 0.018 |
| Llama 3.1 8B | mixed CPT ep2 | – | – | – | 0.439†‡§ | 0.122†‡ |
| Llama 3.1 8B | base on NF4 weights (control) | – | – | – | 0.158 | 0.000 |

Adapters are served on the NF4-dequantized base they were trained on (QLoRA). † differs from base, ‡ mixed CPT differs from QA-only at the same epoch, § differs from the same adapter served on the bf16 base (paired bootstrap, Holm-adjusted p < 0.05). '–' = not measured.
