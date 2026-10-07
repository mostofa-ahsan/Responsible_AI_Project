**Table 7. Held-out documents: adapters served on their training base**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.642 | 0.518 | 0.088 | 0.259 | 0.000 |
| Qwen3-8B | concise base (≤ 40 words) | 0.547† | 0.347† | 0.097 | 0.291† | **0.004** |
| Gemma 4 E4B | base | **0.679** | **0.541** | **0.065** | 0.177 | 0.000 |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.564† | 0.343† | 0.068 | 0.276† | 0.000 |
| Llama 3.1 8B | base | 0.534 | 0.493 | 0.100 | 0.179 | 0.000 |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.506 | 0.375† | 0.102 | **0.311†** | 0.000 |

Adapters are served on the NF4-dequantized base they were trained on (QLoRA). † differs from base, ‡ mixed CPT differs from QA-only at the same epoch, § differs from the same adapter served on the bf16 base (paired bootstrap, Holm-adjusted p < 0.05). '–' = not measured.
