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
python -m pytest tests -q            # all tests
python -m pytest tests/test_parse.py -k bibliography   # single test
python src/audit_hyphenation.py      # list suspicious line-break joins (logs/hyphenation_audit_report.txt)
```

`--reclean` with no `--doc` processes every doc and runs Docling for any doc without a Docling cache entry. Use `--doc` to limit it.

Tests use synthetic `{label, text, page}` items with `parse.clean_items()`, so they don't need Docling or the PDFs. `tests/conftest.py` puts `src/` on `sys.path`. There is no lint config.

## Project status

Phases 1 (inventory) and 2 (parse) are done. Phase 3 (chunking) is next. `closed_book_baseline` in `config.yaml` is still `TODO` (it will be the local base model to be fine-tuned), and `src/llm.py` isn't built yet.

Inventory facts (`data/inventory.csv`):
- `doc_id` is the stable key for every later stage. It's a slug of the cleaned filename; if two files would get the same slug, a short sha256 suffix is added.
- 100 PDFs: 39 books (9,138 pages) and 61 articles (1,023 pages). None are corrupt. All have a text layer except `Books_AI/bulk-download/Frontmatter.pdf`, which is partial and gets OCR.
- No duplicates. The highest pairwise text overlap is about 4%. Similar filenames such as `Artificial Intelligence .pdf` and `Artificial Intelligence.pdf` are different books, so don't merge by filename.
- `Books_AI/bulk-download/` is one book, *AI Horizons*, split into 10 chapter files. Its files share a `doc_group`, and the train/test split must keep them together.

Parsing (`src/parse.py`, Docling for both books and articles; GROBID was deliberately not used):
- Docling labels every heading as level 1. `heading_levels()` infers depth from numbering, ALL-CAPS style and "Chapter N" headings, and `section_path` is built from that.
- Non-content blocks are kept but tagged `skip: true` with a `skip_reason` (references, index, toc, front_matter, back_matter, copyright, footnote). Downstream stages must filter on `skip`. Running headers and footers, and page numbers, are dropped.
- `data/parsed/metadata.csv` is the metadata to use downstream (title/authors/year with `*_source` columns). Rows that need a human check are in `metadata_review.csv`. Bulk-download chapters get the title "AI Horizons: <chapter>".
- When you change cleaning rules, bump `PARSER_VERSION` and run `--reclean`.

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
