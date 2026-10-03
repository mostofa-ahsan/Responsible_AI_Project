# Run 2026-10-03: QLoRA fine-tuning of three 8B-class models, closed-book evaluation

## Finding

**Closed-book fine-tuning improves format metrics but lowers judge-scored accuracy.**

All three models were fine-tuned on 6,680 source-grounded QA pairs (with paraphrases, 19,974 training examples). For every model and every test set:
- token F1 and ROUGE-L rose by about 10–20 points;
- Opus-judged accuracy fell, by 8.5–15 points on questions about facts the model never saw.

| Δ fine-tuned − base (paired 95% CI) | Qwen3-8B | Gemma 4 E4B | Llama 3.1 8B |
|---|---|---|---|
| Judge accuracy, test_indomain (unseen chunks) | **−0.114** [−0.144, −0.085] | **−0.150** [−0.180, −0.119] | **−0.085** [−0.114, −0.055] |
| Judge accuracy, test_heldout_docs (unseen documents) | **−0.136** [−0.164, −0.109] | **−0.149** [−0.178, −0.120] | **−0.098** [−0.126, −0.068] |
| Judge accuracy, test_seen_facts (training facts, new wording) | **−0.077** [−0.118, −0.036] | **−0.104** [−0.149, −0.061] | −0.018 [−0.059, 0.023] |
| Token F1, test_heldout_docs | +0.096 | +0.180 | +0.164 |
| ROUGE-L, test_heldout_docs | +0.091 | +0.154 | +0.130 |

**Mechanism.** Fine-tuning taught the dataset's answer *style*: short, specific, confident answers (about 31–34 words, against 120–230 for the base models). On questions whose facts the model doesn't hold, the style becomes a confident list of plausible but wrong specifics. The base model's long, broad answers often happen to include the key items, so the judge gives them partial or full credit. Most of the drop is correct → partial and partial → incorrect.

A second, smaller failure mode is answers that turn generic, defining the topic instead of naming the specific items asked for.

**Seen facts don't rescue it.** test_seen_facts asks training questions in new wording (279 questions, each paraphrase checked by the judge). Even there, one epoch of QLoRA didn't make the facts recallable: Qwen and Gemma still decline, and Llama is flat. Closed-book supervised fine-tuning here changes *how* the model answers far more than *what it knows*.

The unsupported-claims count (test_seen_facts only, measured against the short reference) is about 90% "2+" for the base models and about 40% "2+" for the fine-tuned ones. That reflects answer length, since long answers make more claims beyond a short reference, not a rise in fabrication.

### Three examples (Qwen3-8B; the base answer was judged correct, the fine-tuned answer incorrect)

1. **test_indomain.** *Which issues does the Society at Large category cover in the framework of GenAI threats in higher education?*
   - Reference: ethics, equity, and the economy, beyond individual classrooms and institutions.
   - Fine-tuned: "The Society at Large category covers issues such as privacy, security, and the potential for GenAI to be used for malicious purposes."
   - Base: a 193-word list that leads with ethical and moral concerns (bias, fairness), among others.
2. **test_indomain.** *Which kinds of university systems do DoS and DDoS attacks often target?*
   - Reference: online learning platforms and research databases.
   - Fine-tuned: "…university systems that rely on legacy infrastructure, such as outdated firewalls, weak access controls, and insufficient monitoring tools."
   - Base: a 121-word list starting with learning management systems (Moodle, Blackboard, Canvas) and email servers.
3. **test_heldout_docs.** *What topics should policymakers fund research on to advance responsible AI?*
   - Reference: bias detection, fairness and transparency, plus collaboration with academic and industry experts.
   - Fine-tuned: "…the societal impact of AI, including its effects on employment, privacy, and social equity."
   - Base: a 199-word list starting with bias and fairness, then transparency and explainability.

### Recommendation: run the RAG arms next

The test questions ask for specific, source-bound facts, so a closed-book model can't answer them reliably, and fine-tuning alone makes it more confidently wrong. The next experiment should be the planned **base + RAG** and **fine-tuned + RAG (RAFT format: evidence in context, cited answer)** arms, using the retrieval setup recommended earlier: BM25 + dense (no header) fused with reciprocal-rank fusion, reranked by Qwen3-Reranker-4B, recall@5 92%. The fine-tuned model's concise, grounded style should pay off when the evidence is in context. That's the comparison the paper needs.

## What ran

| Step | Result |
|---|---|
| Dataset (full_v1) | Done earlier today: 9,350 pairs; split, review export (Downloads) and dataset card all in place |
| Downloads | Qwen/Qwen3-8B, google/gemma-4-E4B-it, meta-llama/Llama-3.1-8B-Instruct: all OK (Llama access approved) |
| Smoke tests (30 steps) | All OK. Loss 4.07→1.81 (Qwen), 4.44→1.98 (Gemma), 2.60→1.94 (Llama); 0 masking mismatches |
| Epoch choice | **1 epoch for all models**: a 2-epoch total of 6.64 h exceeded the 4.5 h rule |
| Training | All OK. 1,249 steps each, 66–69 min, peak VRAM 11.4 / 19.8 / 10.3 GB, adapters 172–186 MB |
| Generation | All 18 runs (3 models × 2 arms × 3 splits) via vLLM, with LoRA for fine-tuned; the HF fallback was never needed |
| Grading | Opus 5.5 via Message Batches. Fixed stratified subset of 500 questions per split, identical for every model and arm, no shrinking needed. All 279 test_seen_facts questions graded |
| Report | `comparison.md` / `.csv` (with by-q_type and by-dimension Δ), `judge_accuracy.png`, `environment.txt` |

Per model: `results/run_2026-10-03/<model>/{base,finetuned}/<split>_{predictions,grades,summary,generation}`. Adapters and training logs are in `models/<model>/` (`adapter/`, `train_loss.csv`, `train_summary.json`). No merged full-weight copies were saved.

**Skipped or changed (all logged in `logs/UNATTENDED_DECISIONS.md`):**
- **Disk:** deleted the cached Qwen3-Embedding-4B and Qwen3-Reranker-4B (16 GB) to keep ≥ 10 GB free with both new models. Both are re-downloadable for the RAG arms. 16 GB is free now.
- **unsupported_claims:** added for test_seen_facts only. Adding it to the other splits would have changed the judge schema and invalidated their cached grades.
- **Post-training quantization:** not done (disk). AWQ and GGUF export will be done for the best model only, once the RAG arms decide which is best.
- **Early grading:** Qwen's and Gemma's runs were graded while later models trained. Same subset, same judge, cached.

## Spend vs the $200 cap

| Item | USD |
|---|---|
| full_v1 dataset (Message Batches) | 149.27 |
| Eval grading, test_indomain + test_heldout_docs (all models and arms, plus the earlier base runs) | 22.74 |
| test_seen_facts (build + grading) | 7.26 |
| Pilots, tests, judge comparisons (earlier today) | 11.21 |
| **Total** | **190.47** (cap 200; eval grading 30.00 of its 45 cap) |

## Caveats

- **Same-family judge.** The judge (Opus 5.5) and the data generator (Sonnet 5.5) are from the same family. No human calibration yet; `data/qa_pairs/full_v1_review.xlsx` is pending.
- **Possible length bias.** Judge partial credit may favour long answers that cover more items. A length-controlled re-grade (for example, base answers truncated to fine-tuned length) would bound this effect.
- **One epoch only.** It's possible that more epochs (or a higher learning rate) would let the models recall the seen facts, but the seen-facts result suggests style dominates at this scale.

## Next steps

1. **RAG arms** (above): re-download the embedder and reranker, add retrieval to `eval_closedbook.py`, train a RAFT-format adapter for one model, and compare the four arms.
2. **Length-controlled judging** and **human calibration** of the judge (the review workbook).
3. **Multi-hop and unanswerable questions**, so models learn to abstain instead of answering confidently.
4. **Knowledge-injection probe:** 2–3 epochs for one model on test_seen_facts only, to test whether recall improves at all.
5. **AWQ/GGUF export** of the best model once the RAG comparison picks it.
