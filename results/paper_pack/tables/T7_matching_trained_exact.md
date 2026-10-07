**Table 7. Exact training questions: adapters served on their training base**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.657 | 0.559 | 0.093 | 0.266 | 0.000 |
| Qwen3-8B | concise base (≤ 40 words) | 0.537† | 0.337† | 0.096 | 0.320† | **0.004** |
| Gemma 4 E4B | base | **0.702** | **0.562** | **0.073** | 0.173 | 0.000 |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.546† | 0.316† | 0.082 | 0.289† | 0.002 |
| Llama 3.1 8B | base | 0.552 | 0.517 | 0.100 | 0.179 | 0.000 |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.478† | 0.369† | 0.117 | **0.324†** | 0.000 |

Adapters are served on the NF4-dequantized base they were trained on (QLoRA). † differs from base, ‡ mixed CPT differs from QA-only at the same epoch, § differs from the same adapter served on the bf16 base (paired bootstrap, Holm-adjusted p < 0.05). '–' = not measured.
