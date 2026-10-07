**Table 7. Exact training questions: adapters served on their training base**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.657 | 0.559 | 0.093 | 0.266 | 0.000 |
| Qwen3-8B | concise base (≤ 40 words) | 0.537 | 0.337† | 0.096 | 0.320 | 0.004 |
| Qwen3-8B | QA-only ep1 | 0.590 | 0.402† | 0.061 | 0.453 | 0.068 |
| Qwen3-8B | QA-only ep2 | 0.761§ | 0.640†§ | 0.028† | 0.704§ | 0.496§ |
| Qwen3-8B | QA-only ep3 | 0.835§ | 0.696†§ | 0.021†§ | 0.767§ | 0.610§ |
| Qwen3-8B | mixed CPT ep1 | 0.568 | 0.358†‡ | 0.073 | 0.413‡ | 0.032‡ |
| Qwen3-8B | base on NF4 weights (control) | 0.645 | 0.544 | 0.106 | 0.265 | 0.000 |
| Gemma 4 E4B | base | 0.702 | 0.562 | 0.073 | 0.173 | 0.000 |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.546 | 0.316† | 0.082 | 0.289 | 0.002 |
| Gemma 4 E4B | QA-only ep1 | 0.566 | 0.376† | 0.075 | 0.428 | 0.042 |
| Gemma 4 E4B | QA-only ep2 | 0.771§ | 0.618§ | 0.029†§ | 0.677§ | 0.418§ |
| Gemma 4 E4B | QA-only ep3 | 0.842§ | 0.688†§ | 0.019†§ | 0.764§ | 0.590§ |
| Gemma 4 E4B | mixed CPT ep1 | 0.516‡ | 0.330†‡ | 0.067 | 0.391‡ | 0.012‡ |
| Gemma 4 E4B | base on NF4 weights (control) | 0.691 | 0.544 | 0.063 | 0.188 | 0.000 |
| Llama 3.1 8B | base | 0.552 | 0.517 | 0.100 | 0.179 | 0.000 |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.478 | 0.369† | 0.117 | 0.324 | 0.000 |
| Llama 3.1 8B | QA-only ep1 | 0.636 | 0.460 | 0.050† | 0.510 | 0.136§ |
| Llama 3.1 8B | QA-only ep2 | 0.963§ | 0.823†§ | 0.008† | 0.943§ | 0.900§ |
| Llama 3.1 8B | QA-only ep3 | **0.986§** | **0.851†§** | **0.003†§** | **0.974§** | **0.954§** |
| Llama 3.1 8B | mixed CPT ep1 | 0.574‡ | 0.418† | 0.056† | 0.446‡§ | 0.040‡ |
| Llama 3.1 8B | mixed CPT ep2 | 0.809‡§ | 0.669†‡§ | 0.017† | 0.731‡§ | 0.510‡§ |
| Llama 3.1 8B | base on NF4 weights (control) | 0.553 | 0.514 | 0.085 | 0.192 | 0.000 |

Adapters are served on the NF4-dequantized base they were trained on (QLoRA). † differs from base, ‡ mixed CPT differs from QA-only at the same epoch, § differs from the same adapter served on the bf16 base (paired bootstrap, Holm-adjusted p < 0.05). '–' = not measured.
