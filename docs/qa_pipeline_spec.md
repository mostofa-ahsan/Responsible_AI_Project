# QA Mining Pipeline — Spec for Claude Code

Project: Responsible_AI_Project
Goal: Mine high-quality, source-grounded QA pairs from ~100 books and journal articles on responsible AI in higher education, for fine-tuning and a four-way evaluation (base, base+RAG, fine-tuned, fine-tuned+RAG).

## Ground rules
- Python 3.12, always use `.venv`. Add every dependency to `requirements.txt`.
- All code lives in `src/`; config in `config.yaml`; never hard-code paths, model names, or keys.
- **Never modify `data/raw_pdfs/`.** Read-only.
- Every stage reads from the previous stage's output folder and writes JSONL/CSV to its own folder. Stages must be **resumable**: skip items already processed (track by id), so a crash mid-run doesn't redo work.
- Log to `logs/<stage>.log`; print a summary (counts, failures, cost estimate) at the end of each stage.
- Work in phases. **Stop at the end of each phase and report results before starting the next.**

## LLM client (src/llm.py)
- One wrapper with a `complete(prompt, system, model, json_schema=None)` function supporting:
  - `anthropic` provider (reads `ANTHROPIC_API_KEY` from `.env`)
  - `openai_compatible` provider (base_url + key; covers vLLM / Ollama / other hosted endpoints)
- Retries with exponential backoff, rate limiting, JSON-mode output with validation (pydantic), and token/cost tracking per call.
- Disk cache keyed on (model, prompt hash) so re-runs don't re-pay.
- `config.yaml` holds separate model settings for: `density_scorer` (cheap), `generator`, `judge` (must differ from generator to avoid self-preference), `closed_book_baseline`.

---

## Phase 1 — Inventory (src/inventory.py)
Recursively scan `data/raw_pdfs/`. Output `data/inventory.csv`:
`doc_id, path, folder (book|article), size_mb, pages, text_layer (yes|partial|no), title, author, year, sha256, duplicate_of`
- Text layer: sample pages, check extractable characters per page.
- Duplicates: identical sha256, plus near-duplicate titles (fuzzy match) — pay attention to `Books_AI/bulk-download/`.
- `doc_id` = short stable slug of the filename.
**Report:** totals, OCR-needed files, duplicates, corrupt files. STOP.

## Phase 2 — Parse (src/parse.py)
- Journal articles → GROBID if available, otherwise Docling. Books → Docling (or Marker). OCR for `text_layer = no`.
- Output one JSON per doc in `data/parsed/`: ordered list of blocks `{block_id, type (heading|paragraph|table|caption|list), text, section_path, page}`.
- Clean: remove running headers/footers, page numbers, fix hyphenation, drop references/bibliography, index, TOC, copyright, acknowledgements (tag them as `skip` rather than deleting).
**Report:** per-doc block counts, failures, 3 random parsed samples for eyeballing. STOP.

## Phase 3 — Chunk (src/chunk.py)
- Structure-aware: group blocks by section, target 500–1,000 tokens, ~100-token overlap, never split mid-paragraph.
- Output `data/chunks/chunks.jsonl`: `{chunk_id, doc_id, title, section_path, page_start, page_end, text, n_tokens}`.
- Density score each chunk (cheap model or heuristic: definitions, claims, numbers, findings) → `density` 0–1. Mark `mine = true` for non-skip chunks ≥150 tokens with density above a configurable threshold.
**Report:** chunk count, token totals, % selected for mining. STOP.

## Phase 4 — QA mining PILOT (src/qa_extract.py, src/qa_generate.py, src/qa_filter.py)
Run on **20 chunks from 5 documents (mix books and articles)** only.

**Stage 1 – Knowledge units** (generator model): from each chunk, extract atomic units
`{unit, type (definition|claim|finding|framework|recommendation|causal), evidence (verbatim span from chunk), doc_id, section_path, page}`.
Validate that `evidence` is an exact substring of the chunk; discard otherwise.

**Stage 2 – Questions** (generator model), target mix:
factual 25%, definition/concept 20%, explanation 20%, comparison 10%, multi-hop 10%, application 5%, unanswerable 10%.
- Answers: 2–5 sentences, only from evidence, end with a citation `(Title, p. X)`.
- Questions must be standalone (no "this chapter", "the author above").
- 2 paraphrases per question, sharing a `group_id`.
- Multi-hop: pair related chunks from different docs via embedding similarity; question must require both.
- Unanswerable: plausible on-topic questions; answer = "The provided sources do not address this." Confirm via retrieval that no chunk supports an answer.
- Tag each pair with a readiness `dimension`: governance, leadership_strategy, faculty_readiness, student_ai_literacy, teaching_learning, assessment, privacy_security, equity_accessibility, procurement_technology, monitoring_improvement, general.

**Stage 3 – Filters** (judge model + embeddings):
1. Grounding: judge scores answer support vs evidence 1–5; keep ≥4.
2. Standalone: judge yes/no.
3. Closed-book: ask `closed_book_baseline` with no context; judge correctness → `closed_book_correct` (keep, but record).
4. Retrieval: source chunk must be in top-5 when searching with the question (embedding index over all chunks).
5. Dedup: drop pairs with cosine similarity >0.92 to an existing pair.

Output `data/qa_pairs/pilot.jsonl`, one row per pair:
`qa_id, group_id, question, paraphrases, answer, evidence, doc_id, section_path, page, q_type, dimension, difficulty (easy|medium|hard), closed_book_correct, judge_scores, passed_filters, reject_reason`.
Also write `data/qa_pairs/pilot_review.csv` (human-readable) with an empty `human_verdict` column.

**Report:** pairs generated, pass rate per filter, q_type and dimension distribution, cost, and 10 random examples (5 passed, 5 rejected with reasons). STOP for human review.

## Phase 5 — Full run (after pilot approval)
Same pipeline over all `mine = true` chunks, using batch APIs where available. Then:
- Splits (`src/split.py`): hold out 10–15 docs entirely (stratified by book/article and dimension) + in-domain held-out questions; keep `group_id`s together. Output `data/splits/{train,val,test_indomain,test_heldout_docs}.jsonl`.
- Export a 500-question candidate gold set for human verification.

## Definition of done per phase
Code runs end-to-end from a clean shell, is resumable, has a `--limit` flag for quick tests, and the phase report is printed and saved to `logs/`.
