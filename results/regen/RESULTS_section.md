## Serving precision: adapters on their training base

QLoRA adapters are trained on an NF4-quantized base. Served on the original bf16 base (all earlier results), they lose part of what they learned (results/diagnostics/format_check.md). All adapters were re-served on the NF4-dequantized base (verified per model: answer loss within 2% of NF4) with prompts pre-tokenized by the training chat function. These are the main numbers below; the bf16-served results are kept as an ablation (Table 11).
- Qwen3-8B: answer loss NF4 0.0718 vs dequantized 0.0719 (0.17%); best QA-only epoch ep3, best mixed-CPT epoch None.
- Gemma 4 E4B: answer loss NF4 0.0524 vs dequantized 0.0517 (1.31%); best QA-only epoch ep3, best mixed-CPT epoch None.
- Llama 3.1 8B: answer loss NF4 0.0115 vs dequantized 0.0114 (1.49%); best QA-only epoch ep2, best mixed-CPT epoch None.
- Exact training questions (judge accuracy): Qwen3-8B: base 0.657, QA-only ep3 0.835; Gemma 4 E4B: base 0.702, QA-only ep3 0.842; Llama 3.1 8B: base 0.552, QA-only ep2 0.963.
- Reworded training questions (judge accuracy): Qwen3-8B: base 0.649, QA-only ep3 0.619; Gemma 4 E4B: base 0.668, QA-only ep3 0.613; Llama 3.1 8B: base 0.522, QA-only ep2 0.719.
- In-domain (unseen chunks) (judge accuracy): Qwen3-8B: base 0.669, QA-only ep3 0.494; Gemma 4 E4B: base 0.731, QA-only ep3 0.496; Llama 3.1 8B: base 0.575, QA-only ep2 0.501.
- Held-out documents (judge accuracy): Qwen3-8B: base 0.642, QA-only ep3 0.431; Gemma 4 E4B: base 0.679, QA-only ep3 0.449; Llama 3.1 8B: base 0.534, QA-only ep2 0.452.
