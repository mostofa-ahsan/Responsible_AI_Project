**Table 7. Reworded training questions: adapters served on their training base**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.649 | 0.523 | 0.108 | 0.213 | **0.000** |
| Qwen3-8B | concise base (≤ 40 words) | 0.582† | 0.316† | 0.105 | 0.263† | **0.000** |
| Gemma 4 E4B | base | **0.668** | **0.530** | **0.063** | 0.151 | **0.000** |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.595† | 0.303† | 0.077 | 0.248† | **0.000** |
| Llama 3.1 8B | base | 0.522 | 0.498 | 0.083 | 0.154 | **0.000** |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.530 | 0.356† | 0.115 | **0.265†** | **0.000** |

Adapters are served on the NF4-dequantized base they were trained on (QLoRA). † differs from base, ‡ mixed CPT differs from QA-only at the same epoch, § differs from the same adapter served on the bf16 base (paired bootstrap, Holm-adjusted p < 0.05). '–' = not measured.
