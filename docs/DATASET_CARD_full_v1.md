# Dataset card: RAIHE-QA full_v1

Source-grounded question–answer pairs about responsible AI in higher education, mined from books and journal articles, for fine-tuning and evaluating a domain model (closed-book, RAG, and fine-tuned + RAG arms).

**Status (2026-10-03):** generated, filtered and split. Human review of a 130-pair sample is pending. All quality numbers below come from automatic filters with an LLM judge.

## Summary

| | |
|---|---|
| Source corpus | 100 PDFs: 39 books (9,138 pages), 61 journal articles (1,023 pages) |
| Chunks mined | 2,432 (of 6,406), with 2,326 used for generation |
| Pairs generated / kept | 11,630 / **9,350** (80.4%) |
| Training questions incl. paraphrases | 27,965 (9,253 pairs with 2 paraphrases, 91 with 1) |
| Question types | factual 25.4%, definition 26.5%, explanation 26.8%, application 21.3% |
| Difficulty | easy 33.9%, medium 44.9%, hard 21.2% |
| Splits | train 6,680 · val 393 · test_indomain 791 · test_heldout_docs 1,486 (10 held-out documents) |
| Generation cost | $149.27 (Message Batches API) |

## Source corpus

- 100 PDFs on responsible AI, AI in education and higher-education policy: 39 books or book chapter sets and 61 peer-reviewed articles, all with a text layer. One book (*AI Horizons*) is split into 10 chapter files and treated as one document (`doc_group`) throughout.
- No duplicates: the highest pairwise text overlap is about 4%. Documents with similar file names are different works.
- The documents are publisher-licensed. The dataset contains verbatim evidence spans, so it is for internal research use and should not be redistributed.

## Pipeline

| Stage | Method | Model |
|---|---|---|
| Parse | Docling (layout + tables) on GPU. Section hierarchy rebuilt from numbering and chapter detection. References, index, TOC, front and back matter, copyright and footnotes tagged and excluded. | — |
| Chunk | 500–1,000 tokens (Qwen3-8B tokenizer), never across a chapter or top-level section, never mid-paragraph, about 100-token sentence overlap | — |
| Select | Heuristic density ≥ 0.35, domain topic score ≥ 2 hits/100 words, ≥ 150 tokens, not appendix or table rows (> 60% table characters), and some AI content (ai_score > 0) | — |
| Extract knowledge units | Atomic units (definition, claim, finding, framework, recommendation, causal) with a **verbatim** evidence span of complete sentences. Units whose evidence is not an exact substring of the chunk, or ends mid-sentence, are discarded (503 of 16,003). Review methodology is skipped. | claude-sonnet-5-5 |
| Generate | 5 pairs per chunk from the valid units only, with slots allocated to the type and difficulty mixes. Answers restate only the evidence. Questions must be standalone (banned-phrase list). Each question gets 2 paraphrases. The citation (title, pages) is stored separately from the answer. | claude-sonnet-5-5 |
| Filter | Regex pre-filter, LLM judge, answer-length rule, one repair pass, paraphrase regeneration, embedding dedup (see below) | judge: claude-opus-5-5 |

All LLM calls ran through the Message Batches API (50% price), cached by request.

## Filters and thresholds

1. **Pre-filter (regex):** questions or paraphrases that presuppose a source ("the evidence", "this study", "according to the …", "is described as", "the following", …), and meta-text referring to the evidence or unit IDs.
2. **Judge (claude-opus-5-5), one call per pair:**
   - *grounding* 1–5: does the answer restate only what the evidence says? Keep if ≥ 4 (inference or synthesis caps at 3).
   - *standalone*: understandable without the source.
   - *value* 1–5: domain knowledge worth learning. Keep if ≥ 3. Methodology, inclusion criteria, sample or region trivia, course logistics, table rows and circular answers (which only restate the question) score ≤ 2.
   - *question_well_formed*: grammatical and natural, not garbled, self-answering or contrived.
   - *paraphrase equivalence*, per paraphrase.
3. **Answer length:** 1–5 sentences for factual and definition questions, 2–5 for explanation and application (1–5 for any type when the evidence is a single sentence).
4. **Repair:** pairs failing grounding, standalone, meta-text or well-formedness go back to the generator once with the judge's reason, then are re-judged with the same rubric. Low-value pairs are not repaired.
5. **Paraphrases:** non-equivalent paraphrases are regenerated once and re-checked, and dropped if they still fail. The pair itself is kept.
6. **Dedup** (Qwen3-Embedding-4B): drop a pair if its question has cosine > 0.92 to a kept pair anywhere, or its answer has cosine > 0.90 to a kept pair from the same document.

## Pass rates (full_v1)

| | Pairs | Share of generated |
|---|---|---|
| Passed on the first judge pass | 8,080 | 69.5% |
| Repair attempted | 2,004 | 17.2% |
| Passed after repair | 1,700 | 84.8% of attempts |
| **Final** | **9,350** | **80.4%** (17.2% of kept pairs were repaired) |

Final reject reasons (a pair can have several): low value 1,637 (14.1%), not standalone 492 (4.2%), duplicate 430 (3.7%), grounding 149 (1.3%), garbled question 95 (0.8%), answer length 50 (0.4%), meta-text 1.

## Distributions (kept pairs)

| Readiness dimension | Pairs | Share |
|---|---|---|
| teaching_learning | 1,794 | 19.2% |
| governance | 1,572 | 16.8% |
| general | 1,097 | 11.7% |
| equity_accessibility | 871 | 9.3% |
| student_ai_literacy | 808 | 8.6% |
| privacy_security | 639 | 6.8% |
| monitoring_improvement | 634 | 6.8% |
| faculty_readiness | 627 | 6.7% |
| leadership_strategy | 514 | 5.5% |
| assessment | 459 | 4.9% |
| procurement_technology | 335 | 3.6% |

Per-document counts are in `logs/full_v1_final_report.txt`.

## Splits

- **test_heldout_docs:** 10 whole documents (6 books, 4 articles; *AI Horizons* as one unit), 1,486 pairs, 16%. They're chosen to keep books and articles in proportion and to cover every dimension.
- **train / val / test_indomain:** the remaining pairs, split **by chunk**, so no evidence passage appears in two splits. That gives 6,680 / 393 / 791 pairs.
- **Checks (all pass):** no chunk or `group_id` (question plus its paraphrases) appears in two splits; held-out documents appear only in test_heldout_docs; every readiness dimension appears in both test sets.
- One gap: faculty_readiness is thin in test_heldout_docs (about 1%) because few held-out documents cover it.

## Cost

| Stage | Calls | Cost (batch) |
|---|---|---|
| Extract (Sonnet) | 2,401 | $19.11 |
| Generate (Sonnet) | 2,326 | $23.15 |
| Filter: judge (Opus) | 16,703 | $90.05 |
| Filter: repair and paraphrases (Sonnet) | 5,574 | $16.96 |
| **Total** | | **$149.27** ($0.016 per kept pair) |

Pilots and tests before the full run cost about $10 more.

## Known limitations

- **Same-family judge.** The judge (Claude Opus 5.5) and the generator (Claude Sonnet 5.5) are from the same model family, which can favour the generator's style. A Haiku judge agreed with Opus on only 64% of pass/fail decisions (kappa 0.16), so judge choice matters. No human calibration yet (see "Pending").
- **Repaired pairs (17% of the data)** were rewritten once and then passed by the same judge that rejected them, which makes them more likely to contain judge-pleasing errors.
- **Single-chunk questions only.** No multi-hop, comparison or unanswerable questions yet, so the data does not teach the model to abstain.
- **Generated questions reuse source wording.** Retrieval looks easier than it will be for real users (BM25 is unusually strong: recall@5 84% vs 75% for dense retrieval).
- **Uneven coverage.** procurement_technology, assessment and leadership_strategy each make up less than 6%.
- **Heuristic selection.** Density and topic gates miss some valuable narrative text; a Haiku re-score of the borderline band would have added 411 chunks.
- **Metadata.** 30 documents have heuristic or missing author fields. Citations use title and pages only.

## Pending

- Human review of a stratified 130-pair sample (`data/qa_pairs/full_v1_review.xlsx`: 100 passed, 30 rejected), followed by judge–human agreement (Cohen's kappa).
- Closed-book "already known" flag (`closed_book_correct`) and the retrieval filter (`retrieval_rank`): fields present, values null.
- Multi-hop and unanswerable questions, and the RAG/RAFT training format.

## Files

- Pairs: `data/qa_pairs/full_v1.jsonl` (all 11,630 with filter decisions); splits: `data/splits/*.jsonl`, `split_manifest.json`.
- Reports: `logs/full_v1_final_report.txt`, `logs/split_full_v1_report.txt`, `logs/qa_full_v1_report.txt`.
- Code: `src/qa_extract.py`, `src/qa_generate.py`, `src/qa_filter.py`, `src/full_run.py`, `src/split.py`; config: `config.yaml`; spec: `docs/qa_pipeline_spec.md`.
