# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Build a QA fine-tuning pipeline on ~100 books and journal articles about responsible AI in higher education (`data/raw_pdfs/`: `Books_AI/`, `Research Articles_AI/`):

1. Parse PDFs → `data/parsed/`
2. Structure-aware chunks with doc/section/page metadata → `data/chunks/`
3. LLM-generated QA pairs (factual, conceptual, multi-hop, unanswerable) with source grounding → `data/qa_pairs/`
4. Document-level train/test splits to prevent leakage → `data/splits/`
5. LoRA fine-tune of an open 7–8B model → `models/`
6. Four-way evaluation (base, base+RAG, fine-tuned, fine-tuned+RAG) with LLM-as-judge, RAGAS, and statistical tests → `eval/`
7. Side-by-side comparison UI → `ui/`

The detailed spec for the QA-mining part (phases 1–5, LLM client, output schemas) is [docs/qa_pipeline_spec.md](docs/qa_pipeline_spec.md). Follow it, and **stop at the end of each phase to report before starting the next**.

## Conventions

- Python 3.12; all code lives in `src/`. Add every dependency to `requirements.txt`.
- Always activate `.venv` before running or installing anything.
- Never modify anything under `data/raw_pdfs/`; it is the read-only source corpus.
- Paths, thresholds and model names live in `config.yaml`; API keys go in `.env`. Never hard-code any of these.
- Every stage script:
  - reads the previous stage's output and writes JSONL/CSV to its own folder
  - is resumable (skips already-processed ids)
  - has a `--limit` flag
  - logs to `logs/<stage>.log`
  - saves its report to `logs/<stage>_report.txt`
- Use the helpers in `src/utils.py` (`load_config`, `repo_path`, `get_logger`, `Report`, `slugify`). Scripts import it as a sibling module (`from utils import ...`) and are run as `python src/<stage>.py`.
- The judge model must differ from the generator model.

## Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt      # CUDA torch via the cu128 index (see file header)
python src/inventory.py              # Phase 1: data/raw_pdfs -> data/inventory.csv (~20s, ~1s when cached)
python src/inventory.py --limit 5    # quick test
python src/parse.py                  # Phase 2: PDFs -> data/parsed/<doc_id>.json (~20 min GPU for all; resumes)
python src/parse.py --doc <doc_id>   # one doc
python src/parse.py --reclean        # re-apply cleaning rules from data/cache/docling (~90s, no re-parse)
python src/chunk.py                  # Phase 3: data/parsed -> data/chunks/chunks.jsonl (~90s; resumes per doc)
python src/chunk.py --rescore        # recompute density/topic/mine after changing thresholds in config.yaml
python src/embed.py                  # Phase 3b: embed all chunks -> data/index/ (FAISS; ~15 min GPU; reuses unchanged)
python src/qa_extract.py --run pilot_v2    # Phase 4 stage 1: knowledge units (Sonnet; cached)
python src/qa_generate.py --run pilot_v2   # stage 2: QA pairs from valid units
python src/qa_filter.py --run pilot_v2     # stage 3: prefilter + Opus judge + repair + paraphrase fix + dedup
python src/qa_agreement.py --run pilot_v2  # human_verdict vs filters (kappa, disagreements)
python src/retrieval_study.py        # recall@k for dense/BM25/hybrid/rerank on passed questions
python src/density_llm.py            # Haiku scores the borderline density band (report only)
python src/judge_compare.py          # candidate judge (Haiku) vs Opus on a run's judged pairs
python src/full_run.py --run full_v1           # full QA run via Message Batches (run inside tmux)
python src/full_run.py --run full_v1 --status  # stage, batch status, items, spend, ETA
python src/run_report.py --run full_v1         # final dataset report (local files only)
python src/split.py --run full_v1              # train/val/test splits + leakage/dimension checks (re-seeds up to 5x)
bash scripts/after_filter.sh                   # post-filter pipeline (report, split, review export, dry run,
                                               #   smoke test, base eval, READY page); run in tmux
python src/train_qlora.py --dry-run            # CPU: build + tokenize training data, no model load
python src/train_qlora.py --max-steps 30 --output models/smoke   # GPU smoke test -> logs/smoke_test_summary.json
python src/eval_closedbook.py --dry-run --limit 3
python src/eval_closedbook.py --arm base_closedbook --split test_heldout_docs   # vLLM + Opus batch grading
python src/write_ready.py --run full_v1        # logs/READY_FOR_TRAINING.md
bash scripts/model_runs.sh                     # multi-model QLoRA + closed-book eval (tmux; resumable markers)
bash scripts/seen_facts.sh                     # test_seen_facts follow-up (after model_runs.sh)
python src/model_runs.py report                # rebuild results/run_2026-10-03/comparison.md from saved outputs
python -m pytest tests -q            # all tests
python -m pytest tests/test_parse.py -k bibliography   # single test
python src/audit_hyphenation.py      # list suspicious line-break joins (logs/hyphenation_audit_report.txt)
```

`--reclean` with no `--doc` processes every doc and runs Docling for any doc without a Docling cache entry. Use `--doc` to limit it.

Tests use synthetic `{label, text, page}` items with `parse.clean_items()`, so they don't need Docling or the PDFs. `tests/conftest.py` puts `src/` on `sys.path`. There is no lint config.

## Project status

Phases 1–3 are done (inventory, parse, chunk + embed). Phase 4a (single-chunk QA pilot) is at v2; multi-hop, unanswerable, the closed-book check and the retrieval filter come next. `closed_book_baseline` in `config.yaml` is still `TODO`, and `src/llm.py` isn't built yet.

Inventory facts (`data/inventory.csv`):
- `doc_id` is the stable key for every later stage. It's a slug of the cleaned filename; if two files would get the same slug, a short sha256 suffix is added.
- 100 PDFs: 39 books (9,138 pages) and 61 articles (1,023 pages). None are corrupt, and there are no duplicates (the highest pairwise text overlap is about 4%). Similar filenames such as `Artificial Intelligence .pdf` and `Artificial Intelligence.pdf` are different books.
- `Books_AI/bulk-download/` is one book, *AI Horizons*, split into 10 chapter files that share a `doc_group`. Splits must keep them together.

Parsing (`src/parse.py`, Docling for books and articles; GROBID was deliberately not used):
- Docling labels every heading as level 1. `heading_levels()` / `find_chapter_starts()` rebuild the structure. For books, `section_path[0]` is the chapter, detected from "Chapter N" headings, numbering restarts (edited volumes) or a new "X.1" (monographs). For articles, it is the top-level section.
- Non-content blocks are kept but tagged `skip: true` with a `skip_reason`. Downstream stages must filter on `skip`. References and back matter end at the next non-skip heading; TOC and index regions end at a same-or-higher-level or numbered heading, or at prose.
- `data/parsed/metadata.csv` is the canonical title/authors/year; uncertain rows are in `metadata_review.csv`.
- When you change cleaning rules, bump `PARSER_VERSION`, run `--reclean`, run the tests, and compare kept-word totals in the report. A drop usually means a skip region is swallowing content.

Chunking (`src/chunk.py`) and index (`src/embed.py`):
- Token counts use the `base_model` tokenizer (Qwen/Qwen3-8B). Chunks are 500–1,000 tokens (hard max 1,300) and never cross `section_path[0]` or split a paragraph. Overlap is about 100 tokens of trailing sentences, added only on size-based splits.
- `mine` = at least 150 tokens, heuristic `density` >= `density_threshold`, and keyword `topic_score` >= `min_topic_score` (the topic gate drops off-domain material such as medical imaging or blockchain internals), and the doc is not in `exclude_from_mining`.
- The FAISS index (`data/index/chunks.faiss`, IndexFlatIP on normalized Qwen3-Embedding-4B vectors) covers all chunks, not just mineable ones. Row order is in `ids.json`. Queries must be prefixed with `embed.query_instruction`; `embed.search()` does this.

## Environment

- Python 3.12 virtualenv at `.venv/`. Activate with `source .venv/bin/activate`.
- Runs on WSL2 (Linux).
- `.gitignore` excludes `.venv/`, `data/`, `models/`, `logs/`, `.env`, and `*.zip`. The corpus and any trained weights stay local, so don't assume they exist on another checkout, and don't commit generated data.

## Research context

The project supports a study titled **"Leveraging AI: Advancing an Institutional Responsible AI Readiness Score/Index for Higher Education."** It aims to define, measure, and validate a multidimensional index of how ready a university is to adopt AI responsibly across teaching, learning, and assessment. The source drafts are in `docs/*.docx`: the research questions, the introduction and literature review, and a conference session proposal.

The candidate index dimensions (from `docs/Research Topic and Questions_AI.docx`, drawing on UNESCO, EDUCAUSE, OECD, and NIST) are:
institutional strategy & leadership; AI governance & accountability; faculty & staff readiness; student AI literacy; teaching & learning practices; assessment design; data privacy & cybersecurity; equity & accessibility; technology & procurement; monitoring & continuous improvement.

Any generated content, prompts, or evaluation criteria should stay within this domain and these dimensions.

## Data layout

The `data/` subfolders map to the pipeline steps above:

```
data/raw_pdfs/   ~100 source PDFs (inputs, already populated)
  ├─ Books_AI/               books and book chapters (incl. Books_AI/bulk-download/ split chapters)
  └─ Research Articles_AI/   journal articles, mostly prefixed "JA_"
data/parsed/     extracted text from PDFs
data/chunks/     chunked text
data/qa_pairs/   generated question/answer pairs
data/splits/     train/val/test splits
models/          trained or fine-tuned model artifacts
eval/            evaluation code/results
ui/              front-end/demo
notebooks/       exploration
```

PDF filenames contain spaces, ampersands, curly quotes, and trailing spaces (e.g. `"Leveraging GenAI .pdf"`), so always quote paths and handle unusual characters robustly.

QA mining (`src/llm.py`, `src/qa_*.py`, spec in `docs/qa_pipeline_spec.md`):
- All LLM calls go through `llm.LLM(cfg, stage).complete(prompt, system, role=..., json_schema=PydanticModel)`. Roles, models, effort and pricing are in `config.yaml`. Every call is disk-cached in `data/cache/llm/` and logged to `logs/llm_usage.jsonl`, so re-running a stage is free unless a prompt changes. Changing a prompt or schema invalidates the cache for that stage and costs money: check `logs/llm_usage.jsonl` totals.
- A run (`--run NAME`) is defined under `qa.runs` in config. Its files are `data/qa_pairs/<run>_{chunks.json,units.jsonl,generated.jsonl}`, `<run>.jsonl` (final) and `<run>_review.csv`. The chunk selection is frozen in `<run>_chunks.json` after the first run; delete it to reselect.
- `answer` never contains a citation. `citation` is `{title, pages}` from the evidence pages, and training formats decide whether to append it.
- The review CSV keeps filled `human_verdict`/`human_notes` when it is rewritten (`qa_common.write_review_csv`). Never write it any other way.
- The judge (`claude-opus-5-5`) is the same family as the generator (`claude-sonnet-5-5`). Repaired pairs are re-judged by the same judge, so treat repaired passes with extra suspicion when reviewing.

Batch runs (`src/full_run.py`, `llm.BatchRunner`):
- With `LLM_COLLECT=<file>`, `LLM.complete()` records each uncached request and raises `Deferred` instead of calling the API. `full_run.py` runs each stage script in collect mode, sends the recorded requests as ONE Message Batch, and writes each result into the normal cache (same key) and `llm_usage.jsonl` (with `"batch": true`, at 50%). It then re-runs the stage. A stage is done when a collect pass records nothing; a final normal pass (all cache hits) writes the outputs.
- Batch request mappings live in `data/cache/batches/<batch_id>.json` (`custom_id` -> request, `collected` flag). On restart, uncollected batches are polled and collected before anything new is submitted, so a crash never pays twice. Just re-run the same command.
- After `llm.batch_timeout_min` a batch is cancelled. Unfinished, errored, expired, refused or schema-invalid results get one standard-API retry. Billing or spend-limit errors exit with code 3 and the state saved.
- Refusal fallbacks (`fallbacks: default`) only apply on the standard path; the Batches API rejects that parameter.
- Two 4B models (embedder and reranker) on the 24 GB GPU at once thrash WSL memory and stall both jobs. Run GPU steps (retrieval study, the filter's dedup, training, eval) one at a time.

Budget, eval and training notes:
- Spend caps are in `config.yaml` `budget`. `full_run.py` checks the next batch round's estimated cost against `run_max_usd[run]`. If it would exceed the cap, it finalizes with `LLM_OFFLINE=1`: uncached calls raise `Unavailable`, repairs are skipped (`repair_skipped_budget`) and bad paraphrases dropped. `eval_closedbook.py` grades a 50-answer pilot, measures the real cost per answer, and samples (stratified) to stay within `eval_run_max_usd` (per split) and `eval_max_usd`.
- vLLM lives in a separate venv, `.venv-vllm` (vLLM 0.30 needs torch 2.13; the main venv stays on torch 2.11). `eval_closedbook.py` calls `src/vllm_generate.py` with that interpreter and falls back to HF generate. The machine has the NVIDIA driver but no CUDA toolkit (`nvcc`), so the worker sets `VLLM_USE_FLASHINFER_SAMPLER=0`. Anything that JIT-compiles CUDA code will fail here.
- `train_qlora.py` (transformers 5 / trl 1.14) uses `warmup_steps` with a float ratio; `warmup_ratio` no longer exists. It saves and evaluates val loss every epoch, keeps the best checkpoint (`load_best_model_at_end`), and runs 2 epochs by default (3 via `--epochs` or config).
- In scripts, don't `pgrep -f` a command string that also appears in a tmux session's command: the tmux server keeps that argv. Anchor the pattern to the interpreter (`^[^ ]*python[^ ]* src/...`).

Multi-model results (`results/run_2026-10-03/`; read `SUMMARY.md` first):
- Qwen3-8B, Gemma 4 E4B and Llama 3.1 8B were each QLoRA fine-tuned for 1 epoch with an identical recipe (adapters in `models/<name>/adapter/`). Closed-book fine-tuning raises token F1 and ROUGE-L but LOWERS Opus-judged accuracy on every test split, including `test_seen_facts` (paraphrased training questions). The next experiment is the RAG / RAFT arms.
- Judge accuracy uses the fixed stratified subset in `data/splits/eval_subset.json` (500 per split, the same for every model and arm). Keep it fixed for new arms so results stay comparable.
- The Qwen3 embedder and reranker caches were deleted for disk space; re-download them for RAG work.


Local evaluation (`scripts/run_all_localeval.sh`, `src/local_*.py`; read `results/local_eval/STATUS.md` and `results/paper_pack/RESULTS.md` first):
- Fully offline: every stage imports `local_common`, which asserts `LLM_OFFLINE=1` and makes Anthropic/OpenAI client construction raise. Resumable through `results/local_eval/.done/` markers and the vLLM output caches in `results/local_eval/judge_cache/`; each stage has a time box and a failure never stops the pipeline. The paper pack (S7) always runs last.
- Judge: Mistral-Small-3.2-24B AWQ (fallback Phi-4 AWQ) through `src/vllm_json_worker.py` (guided JSON, `.venv-vllm`). It reuses the Opus rubric verbatim and is calibrated only against the Opus grades of run_2026-10-03 (DEV/TEST split by qa_id).
- Mistral3 checkpoints need `src/vllm_shims/sitecustomize.py` on the worker's PYTHONPATH (vLLM 0.30's pixtral.py imports names that transformers 5.18 removed). `run_worker` sets this.
- The judge weights are deleted after S4 so that Bespoke-MiniCheck-7B fits the disk. Re-download them before re-grading anything new.
- After a crash or reboot, run `bash scripts/resume_after_crash.sh`. It relaunches the GPU watchdog (`logs/gpu_temp.log`; it creates `results/.gpu_pause` at ≥ 84 °C), the local evaluation (tmux `localeval`) and the seed-43 queue (tmux `seeds`, `scripts/seeds_queue.sh`). Checkers append fsync'd JSONL and resume; seed training checkpoints every 150 steps.
- Bespoke-MiniCheck-7B needs two fixes, both applied automatically:
  - The remote-code InternLM2 tokenizer is broken under transformers 5.18, so `local_checks.minicheck_tokenizer()` builds a plain fast-tokenizer folder.
  - vLLM 0.30's `InternLM2ForCausalLM.forward` lacks a default for `intermediate_tensors`; the shim adds it.
