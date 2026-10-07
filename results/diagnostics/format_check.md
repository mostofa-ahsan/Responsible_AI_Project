# Train/inference prompt-format check (QA-only ep3 adapters)

_Generated 2026-10-06 21:24 by `src/diag_format.py` (no training, no result files changed). 20 items of test_trained_exact (= training items, original question), seed 0._

## 1. Training token sequence vs the prompt vLLM received

| Model | items with identical prompt tokens | training-tokenizer prefix mismatches | vLLM finish reasons |
|---|---|---|---|
| qwen3-8b | 20/20 | 0 | stop |
| gemma-4-e4b-it | 0/20 | 0 | stop |
| llama-3.1-8b-instruct | 20/20 | 0 | stop |

### qwen3-8b

Training prompt (decoded, first item):

```
<|im_start|>system
You are an expert on responsible AI in higher education. Answer accurately and concisely.<|im_end|>
<|im_start|>user
What shapes students' adoption of generative AI in a unified socio-cognitive model for engineering education, making their behavioural intentions contextually and socially embedded?<|im_end|>
<|im_start|>assistant
<think>

</think>


```
vLLM prompt (decoded, same item):

```
<|im_start|>system
You are an expert on responsible AI in higher education. Answer accurately and concisely.<|im_end|>
<|im_start|>user
What shapes students' adoption of generative AI in a unified socio-cognitive model for engineering education, making their behavioural intentions contextually and socially embedded?<|im_end|>
<|im_start|>assistant
<think>

</think>


```
Last supervised training tokens: `['Ġembedded', '.', '<|im_end|>', 'Ċ']`; last vLLM output tokens: `['Ġsocially', 'Ġembedded', '.', '<|im_end|>']`.

No token differences in the first 10 items.

### gemma-4-e4b-it

Training prompt (decoded, first item):

```
<bos><|turn>system
You are an expert on responsible AI in higher education. Answer accurately and concisely.<turn|>
<|turn>user
What shapes students' adoption of generative AI in a unified socio-cognitive model for engineering education, making their behavioural intentions contextually and socially embedded?<turn|>
<|turn>model

```
vLLM prompt (decoded, same item):

```
<bos><|turn>system
You are an expert on responsible AI in higher education. Answer accurately and concisely. <turn|>
<|turn>user
What shapes students' adoption of generative AI in a unified socio-cognitive model for engineering education, making their behavioural intentions contextually and socially embedded?<turn|>
<|turn>model

```
Last supervised training tokens: `['▁embedded', '.', '<turn|>', '\n']`; last vLLM output tokens: `['▁behavioural', '▁intentions', '.', '<turn|>']`.

Token differences (first 10 items):
- item 0: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 1: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 2: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 3: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 4: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 5: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 6: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 7: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 8: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])
- item 9: insert training [] (pos [20, 20]) vs vLLM ['▁'] (pos [20, 21])

### llama-3.1-8b-instruct

Training prompt (decoded, first item):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are an expert on responsible AI in higher education. Answer accurately and concisely.<|eot_id|><|start_header_id|>user<|end_header_id|>

What shapes students' adoption of generative AI in a unified socio-cognitive model for engineering education, making their behavioural intentions contextually and socially embedded?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```
vLLM prompt (decoded, same item):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are an expert on responsible AI in higher education. Answer accurately and concisely.<|eot_id|><|start_header_id|>user<|end_header_id|>

What shapes students' adoption of generative AI in a unified socio-cognitive model for engineering education, making their behavioural intentions contextually and socially embedded?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```
Last supervised training tokens: `['Ġsocially', 'Ġembedded', '.', '<|eot_id|>']`; last vLLM output tokens: `['Ġsocially', 'Ġembedded', '.', '<|eot_id|>']`.

No token differences in the first 10 items.

## 2–3. Effect on the QA-only ep3 adapter (HF transformers)

| Model | answer loss: NF4, training prompt | NF4, vLLM prompt | bf16, vLLM prompt | ROUGE-L (5 gens): HF NF4 train / HF NF4 vLLM / HF bf16 vLLM / vLLM | ROUGE-L vLLM fresh (20) | production answers (20) |
|---|---|---|---|---|---|---|
| qwen3-8b | 0.0718 | 0.0718 | 0.1940 | 0.966 / 0.966 / 0.663 / 0.668 | 0.474 | 0.472 |
| gemma-4-e4b-it | 0.0524 | 0.0544 | 0.3780 | 0.841 / 0.841 / 0.512 / 0.503 | 0.367 | 0.381 |
| llama-3.1-8b-instruct | 0.0115 | 0.0115 | 0.0562 | 1.000 / 1.000 / 0.966 / 0.966 | 0.797 | 0.794 |

Generated examples are in `results/diagnostics/format_check.json`.

## 4. Findings

**There is no meaningful prompt-format mismatch.** The training prompt (system prompt, chat template, generation prompt, and for Qwen3 the empty `<think>\n\n</think>\n\n` block) is token-identical to what vLLM receives for Qwen3-8B (20/20 items) and Llama 3.1 8B (20/20; both use the template's fixed 'Today Date: 26 Jul 2024'). The supervised answer ends with the end-of-turn token at which vLLM stops (finish reason 'stop' for every item). Gemma 4 E4B differs by ONE token in every item: vLLM's rendering adds a space ('▁') after the system message (`concisely. <turn|>` vs training `concisely.<turn|>`). Its effect is negligible: answer loss 0.0524 (training prompt) vs 0.0544 (vLLM prompt), identical greedy outputs on 5 items (ROUGE-L 0.841 both).

**The cause is the base-model precision at inference.** The adapters are QLoRA adapters trained on the NF4-quantized base, but every vLLM evaluation applies them to the bf16 base. On identical prompts, answer-token loss (NF4 → bf16):
- qwen3-8b: 0.0718 → 0.1940 (×2.7); greedy ROUGE-L vs the trained answer 0.966 (HF, NF4) → 0.663 (HF, bf16) ≈ 0.668 (vLLM) on 5 items; vLLM over 20 items 0.474 (production answers 0.472).
- gemma-4-e4b-it: 0.0544 → 0.3780 (×6.9); greedy ROUGE-L vs the trained answer 0.841 (HF, NF4) → 0.512 (HF, bf16) ≈ 0.503 (vLLM) on 5 items; vLLM over 20 items 0.367 (production answers 0.381).
- llama-3.1-8b-instruct: 0.0115 → 0.0562 (×4.9); greedy ROUGE-L vs the trained answer 1.000 (HF, NF4) → 0.966 (HF, bf16) ≈ 0.966 (vLLM) on 5 items; vLLM over 20 items 0.797 (production answers 0.794).

Llama's memorization is strong enough to survive the switch (bf16 loss still 0.056); Qwen's and Gemma's is not, which explains low exact-reproduction despite train loss ≈ 0.11. vLLM reproduces HF-bf16 closely, so the engine is not at fault. This matches the earlier deployment result (Llama ep2: NF4 0.962 vs bf16 0.853 judge accuracy on exact questions).

## 5. Proposed fix (not applied)

1. Serve every adapter on the base it was trained on: export, per model, an **NF4-dequantized bf16 checkpoint** (quantize with the training BitsAndBytesConfig, `dequantize_4bit` every 4-bit weight, save bf16) and serve it with vLLM + LoRA. vLLM 0.30 cannot load bitsandbytes NF4 itself. Verified in memory on Qwen3 ep3 (20 items): answer loss 0.0719 with the dequantized base vs 0.0718 NF4 and 0.1940 plain bf16 (`dequant_check.json`).
2. Remove the Gemma space by giving vLLM pre-tokenized prompts rendered with the training tokenizer (`prompt_token_ids` from `train_qlora.chat`) instead of `llm.chat`, for every model.
3. Keep the base and concise-base systems on the original bf16 base (they have no adapter); only Gemma's base/concise answers would change through step 2.

**Cost** (all local, no API cost). Answers to regenerate (eval-subset prompts):
- qwen3-8b: QA-only ep1-3 + 1-epoch run 7,116, arm C epochs (1) 1,779, seed 43 ep1 1,279 → 10,174
- gemma-4-e4b-it: QA-only ep1-3 + 1-epoch run 7,116, arm C epochs (1) 1,779, seed 43 ep1 1,279, base + concise base (system-prompt space only) 3,558 → 13,732
- llama-3.1-8b-instruct: QA-only ep1-3 + 1-epoch run 7,116, arm C epochs (2) 3,558, seed 43 ep1 1,279 → 11,953

- Qwen + Gemma: 23,906 answers. Generation ≈ 30 min (dequantized export ~5 min and vLLM ~5–8 min per model at 1,300–1,900 tok/s measured earlier); judge grading ≈ 72 min at the measured ~5.5 answers/s plus ~5 min for the judge swap; MiniCheck-7B + NLI + Flan-T5 on ~84k (answer, fact) pairs ≈ 25 min; stats + pack ≈ 5 min. **Total ≈ 2.3 h.**
- Recommended for consistency: all three models (35,859 answers), because Llama is affected too (smaller effect). **Total ≈ 3.2 h** (judge ≈ 109 min).
- Disk: each dequantized checkpoint needs 16 GB while it is served (one model at a time; delete after). With the 10 GB reserve this means doing generation while neither the judge nor MiniCheck-7B is on disk, i.e. the same swap order as before: generate → judge → MiniCheck.
- Also affected but not in this estimate: the merged 4-bit (AWQ/RTN) deployment variants, which were merged into the bf16 base before quantization.
