**Table 7. In-domain (unseen chunks): adapters served on their training base**

| Model | System | Judge acc | KF recall | Contra. | F1 | Exact repro. |
|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.669 | 0.564 | 0.096 | 0.265 | **0.000** |
| Qwen3-8B | concise base (≤ 40 words) | 0.548† | 0.365† | 0.096 | 0.305† | **0.000** |
| Gemma 4 E4B | base | **0.731** | **0.574** | 0.078 | 0.177 | **0.000** |
| Gemma 4 E4B | concise base (≤ 40 words) | 0.595† | 0.367† | **0.067** | 0.287† | **0.000** |
| Llama 3.1 8B | base | 0.575 | 0.555 | 0.091 | 0.182 | **0.000** |
| Llama 3.1 8B | concise base (≤ 40 words) | 0.510† | 0.418† | 0.102 | **0.322†** | **0.000** |

Adapters are served on the NF4-dequantized base they were trained on (QLoRA). † differs from base, ‡ mixed CPT differs from QA-only at the same epoch, § differs from the same adapter served on the bf16 base (paired bootstrap, Holm-adjusted p < 0.05). '–' = not measured.
