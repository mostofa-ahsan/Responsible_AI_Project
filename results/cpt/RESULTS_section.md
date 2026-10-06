## Arm C: mixed continued pretraining

## Method

QLoRA on a frozen NF4 base (double quantization, bf16 compute) with a bf16 LoRA adapter on the q, k, v, o, gate, up and down projections:  (alpha = 2r, dropout 0.05), lr 1e-4 cosine over 3 epochs, warmup 3%, effective batch 16, max sequence 1024, seed 42, gradient checkpointing. Each epoch mixes (a) RAW: all chunks of the 80 training documents in document order (chunk overlap removed), packed into 1024-token sequences with EOS between documents, loss on every token, and (b) QA: every training pair as the original question plus one paraphrase (rotating per epoch), answer-only loss; shuffled together per epoch.
- Llama 3.1 8B: 3,547 RAW sequences (3,631,352 tokens) + 13,360 QA examples (1,693,169 tokens, 653,884 answer tokens) per epoch; RAW:QA = 0.68 / 0.32 by tokens, 0.85 / 0.15 by loss tokens (target ≈ 50/50 was not reached with this construction; the data were not altered).
- Qwen3-8B: 3,644 RAW sequences (3,730,635 tokens) + 13,360 QA examples (1,474,399 tokens, 673,106 answer tokens) per epoch; RAW:QA = 0.72 / 0.28 by tokens, 0.85 / 0.15 by loss tokens (target ≈ 50/50 was not reached with this construction; the data were not altered).
- Gemma 4 E4B: 3,667 RAW sequences (3,754,352 tokens) + 13,360 QA examples (1,396,032 tokens, 665,142 answer tokens) per epoch; RAW:QA = 0.73 / 0.27 by tokens, 0.85 / 0.15 by loss tokens (target ≈ 50/50 was not reached with this construction; the data were not altered).

See results/cpt/SUMMARY.md, Table 7 (arm C) and Figures 9–10 (arm C).
