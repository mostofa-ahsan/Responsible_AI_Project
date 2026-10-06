# Arm C: mixed continued pretraining (raw text + QA)

_Generated 2026-10-06 15:03 by `src/cpt_report.py`; numbers from results/cpt/*.csv|json._

## Method

QLoRA on a frozen NF4 base (double quantization, bf16 compute) with a bf16 LoRA adapter on the q, k, v, o, gate, up and down projections: Llama 3.1 8B r=128, Qwen3-8B r=128, Gemma 4 E4B r=64 (alpha = 2r, dropout 0.05), lr 1e-4 cosine over 3 epochs, warmup 3%, effective batch 16, max sequence 1024, seed 42, gradient checkpointing. Each epoch mixes (a) RAW: all chunks of the 80 training documents in document order (chunk overlap removed), packed into 1024-token sequences with EOS between documents, loss on every token, and (b) QA: every training pair as the original question plus one paraphrase (rotating per epoch), answer-only loss; shuffled together per epoch.
- Llama 3.1 8B: 3,547 RAW sequences (3,631,352 tokens) + 13,360 QA examples (1,693,169 tokens, 653,884 answer tokens) per epoch; RAW:QA = 0.68 / 0.32 by tokens, 0.85 / 0.15 by loss tokens (target ≈ 50/50 was not reached with this construction; the data were not altered).
- Qwen3-8B: 3,644 RAW sequences (3,730,635 tokens) + 13,360 QA examples (1,474,399 tokens, 673,106 answer tokens) per epoch; RAW:QA = 0.72 / 0.28 by tokens, 0.85 / 0.15 by loss tokens (target ≈ 50/50 was not reached with this construction; the data were not altered).
- Gemma 4 E4B: 3,667 RAW sequences (3,754,352 tokens) + 13,360 QA examples (1,396,032 tokens, 665,142 answer tokens) per epoch; RAW:QA = 0.73 / 0.27 by tokens, 0.85 / 0.15 by loss tokens (target ≈ 50/50 was not reached with this construction; the data were not altered).

Leakage audit: passed (no held-out document in RAW or QA, no test/val qa_id in QA, no seen-facts question in QA). Questions sharing a word 13-gram with RAW: indomain 8/791, heldout_docs 0/1486, seen_facts 0/279, trained_exact 2/500.

## Results (judge accuracy; key-fact recall in parentheses)

**exact training questions**
- Llama 3.1 8B: base 0.552 (0.517); QA-only ep1 0.607 (0.430); QA-only ep2 0.853 (0.726); QA-only ep3 0.896 (0.751); mixed CPT ep1 0.564 (0.406); mixed CPT ep2 0.735 (0.583)
- Qwen3-8B: base 0.657 (0.559); QA-only ep1 0.593 (0.399); QA-only ep2 0.667 (0.511); QA-only ep3 0.670 (0.529); mixed CPT ep1 0.569 (0.363)
- Gemma 4 E4B: base 0.702 (0.562); QA-only ep1 0.563 (0.365); QA-only ep2 0.614 (0.422); QA-only ep3 0.638 (0.432); mixed CPT ep1 0.534 (0.333)

**paraphrased training questions**
- Llama 3.1 8B: base 0.522 (0.498); QA-only ep1 0.615 (0.387); QA-only ep2 0.674 (0.487); QA-only ep3 0.654 (0.485); mixed CPT ep1 0.572 (0.372); mixed CPT ep2 0.620 (0.419)
- Qwen3-8B: base 0.649 (0.523); QA-only ep1 0.573 (0.338); QA-only ep2 0.590 (0.426); QA-only ep3 0.584 (0.407); mixed CPT ep1 0.575 (0.333)
- Gemma 4 E4B: base 0.668 (0.530); QA-only ep1 0.595 (0.359); QA-only ep2 0.599 (0.386); QA-only ep3 0.629 (0.402); mixed CPT ep1 0.554 (0.317)

**in-domain (unseen chunks)**
- Llama 3.1 8B: base 0.575 (0.555); QA-only ep1 0.525 (0.372); QA-only ep2 0.496 (0.371); QA-only ep3 0.514 (0.371); mixed CPT ep1 0.539 (0.374); mixed CPT ep2 0.528 (0.373)
- Qwen3-8B: base 0.669 (0.564); QA-only ep1 0.535 (0.360); QA-only ep2 0.508 (0.381); QA-only ep3 0.536 (0.385); mixed CPT ep1 0.557 (0.372)
- Gemma 4 E4B: base 0.731 (0.574); QA-only ep1 0.553 (0.371); QA-only ep2 0.530 (0.364); QA-only ep3 0.532 (0.369); mixed CPT ep1 0.547 (0.365)

**held-out documents**
- Llama 3.1 8B: base 0.534 (0.493); QA-only ep1 0.460 (0.325); QA-only ep2 0.465 (0.311); QA-only ep3 0.462 (0.308); mixed CPT ep1 0.502 (0.340); mixed CPT ep2 0.469 (0.323)
- Qwen3-8B: base 0.642 (0.518); QA-only ep1 0.506 (0.345); QA-only ep2 0.477 (0.323); QA-only ep3 0.476 (0.337); mixed CPT ep1 0.499 (0.323)
- Gemma 4 E4B: base 0.679 (0.541); QA-only ep1 0.500 (0.322); QA-only ep2 0.498 (0.327); QA-only ep3 0.484 (0.335); mixed CPT ep1 0.533 (0.335)

## Robustness gap (exact − paraphrase judge accuracy, 279 pairs)

| System | exact | paraphrase | gap [95% CI] |
|---|---|---|---|
| gemma-4-e4b-it__base | 0.686 | 0.668 | 0.018 [-0.022, 0.059] |
| gemma-4-e4b-it__concise40 | 0.556 | 0.595 | -0.039 [-0.075, -0.004] |
| gemma-4-e4b-it__cpt_ep1 | 0.520 | 0.554 | -0.034 [-0.068, 0.000] |
| gemma-4-e4b-it__ep1 | 0.541 | 0.595 | -0.054 [-0.095, -0.016] |
| gemma-4-e4b-it__ep2 | 0.586 | 0.599 | -0.013 [-0.054, 0.029] |
| gemma-4-e4b-it__ep3 | 0.622 | 0.629 | -0.007 [-0.047, 0.032] |
| llama-3.1-8b-instruct__base | 0.543 | 0.522 | 0.022 [-0.016, 0.056] |
| llama-3.1-8b-instruct__concise40 | 0.464 | 0.530 | -0.066 [-0.104, -0.029] |
| llama-3.1-8b-instruct__cpt_ep1 | 0.554 | 0.572 | -0.018 [-0.059, 0.020] |
| llama-3.1-8b-instruct__cpt_ep2 | 0.735 | 0.620 | 0.115 [0.073, 0.156] |
| llama-3.1-8b-instruct__ep1 | 0.597 | 0.615 | -0.018 [-0.063, 0.025] |
| llama-3.1-8b-instruct__ep2 | 0.848 | 0.674 | 0.174 [0.131, 0.217] |
| llama-3.1-8b-instruct__ep3 | 0.889 | 0.654 | 0.235 [0.194, 0.274] |
| qwen3-8b__base | 0.647 | 0.649 | -0.002 [-0.045, 0.038] |
| qwen3-8b__concise40 | 0.522 | 0.582 | -0.061 [-0.102, -0.023] |
| qwen3-8b__cpt_ep1 | 0.559 | 0.575 | -0.016 [-0.054, 0.020] |
| qwen3-8b__ep1 | 0.566 | 0.573 | -0.007 [-0.045, 0.032] |
| qwen3-8b__ep2 | 0.654 | 0.590 | 0.065 [0.022, 0.106] |
| qwen3-8b__ep3 | 0.642 | 0.584 | 0.057 [0.014, 0.099] |

Tables: results/paper_pack/tables/T7_cpt_*.md|tex; figures F9_cpt_four_tests, F10_cpt_losses_perplexity; full statistics: results/cpt/metrics_by_system.csv, significance.csv, robustness_gap.csv.
