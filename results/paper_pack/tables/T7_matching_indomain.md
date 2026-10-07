**Table 7. In-domain (unseen chunks): adapters served on their training base**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.669 | 0.564 | 0.096 | 0.265 | 0.000 |
| Qwen3-8B | concise base (≤ 40 words) | 0.548† | 0.365† | 0.096 | 0.305† | 0.000 |
| Qwen3-8B | QA-only ep1 | 0.521† | 0.329† | 0.077 | 0.354† | 0.002 |
| Qwen3-8B | QA-only ep2 | 0.495† | 0.361† | 0.087 | 0.334† | 0.000 |
| Qwen3-8B | QA-only ep3 | 0.494† | 0.365† | 0.090 | 0.331† | 0.000 |
| Qwen3-8B | mixed CPT ep1 | – | – | – | 0.377†‡ | 0.004 |
| Qwen3-8B | base on NF4 weights (control) | – | – | – | 0.265 | 0.000 |
| Gemma 4 E4B | base | **0.731** | **0.574** | 0.078 | 0.177 | 0.000 |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.595† | 0.367† | 0.067 | 0.287† | 0.000 |
| Gemma 4 E4B | QA-only ep1 | 0.541† | 0.353† | **0.065** | 0.365† | 0.006 |
| Gemma 4 E4B | QA-only ep2 | 0.511† | 0.362† | 0.068 | 0.333†§ | 0.002 |
| Gemma 4 E4B | QA-only ep3 | 0.496† | 0.355† | 0.078 | 0.324† | 0.002 |
| Gemma 4 E4B | mixed CPT ep1 | – | – | – | **0.381†‡** | **0.008** |
| Gemma 4 E4B | base on NF4 weights (control) | – | – | – | 0.188† | 0.000 |
| Llama 3.1 8B | base | 0.575 | 0.555 | 0.091 | 0.182 | 0.000 |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.510† | 0.418† | 0.102 | 0.322† | 0.000 |
| Llama 3.1 8B | QA-only ep1 | 0.507† | 0.340† | 0.084 | 0.352† | 0.002 |
| Llama 3.1 8B | QA-only ep2 | 0.501† | 0.350† | 0.085 | 0.330† | 0.002 |
| Llama 3.1 8B | QA-only ep3 | 0.490† | 0.346† | 0.084 | 0.334† | 0.000 |
| Llama 3.1 8B | mixed CPT ep1 | – | – | – | 0.373†‡ | 0.002 |
| Llama 3.1 8B | mixed CPT ep2 | – | – | – | 0.351†‡ | 0.000 |
| Llama 3.1 8B | base on NF4 weights (control) | – | – | – | 0.193† | 0.000 |

Adapters are served on the NF4-dequantized base they were trained on (QLoRA). † differs from base, ‡ mixed CPT differs from QA-only at the same epoch, § differs from the same adapter served on the bf16 base (paired bootstrap, Holm-adjusted p < 0.05). '–' = not measured.
