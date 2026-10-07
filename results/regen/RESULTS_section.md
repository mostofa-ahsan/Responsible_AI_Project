## Serving precision: adapters on their training base

QLoRA adapters are trained on an NF4-quantized base. Served on the original bf16 base (all earlier results), they lose part of what they learned (results/diagnostics/format_check.md). All adapters were re-served on the NF4-dequantized base (verified per model: answer loss within 2% of NF4) with prompts pre-tokenized by the training chat function. These are the main numbers below; the bf16-served results are kept as an ablation (Table 11).
- Qwen3-8B: answer loss NF4 0.0718 vs dequantized 0.0719 (0.17%); best QA-only epoch None, best mixed-CPT epoch None.
- Exact training questions (judge accuracy): Qwen3-8B: base 0.657; Gemma 4 E4B: base 0.702; Llama 3.1 8B: base 0.552.
- Reworded training questions (judge accuracy): Qwen3-8B: base 0.649; Gemma 4 E4B: base 0.668; Llama 3.1 8B: base 0.522.
- In-domain (unseen chunks) (judge accuracy): Qwen3-8B: base 0.669; Gemma 4 E4B: base 0.731; Llama 3.1 8B: base 0.575.
- Held-out documents (judge accuracy): Qwen3-8B: base 0.642; Gemma 4 E4B: base 0.679; Llama 3.1 8B: base 0.534.
