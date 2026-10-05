# Responsible AI in Higher Education: QA dataset & fine-tuning experiments

Pipeline: PDF parsing (Docling) → structure-aware chunking → QA generation with Claude (Sonnet) → judge filters (Opus) → document-level splits → QLoRA fine-tuning of Qwen3-8B, Gemma 4 E4B and Llama 3.1 8B → closed-book evaluation (Opus judge, token F1, ROUGE-L).

- Dataset: 9,350 QA pairs from 100 books/articles (`docs/DATASET_CARD_full_v1.md`)
- Results: `results/run_2026-10-03/SUMMARY.md`
- Headline: 1-epoch closed-book QLoRA raises F1/ROUGE-L but lowers judge-rated accuracy on all three models.

Data, model weights and adapters are not included (source material is copyrighted).
To reproduce: Python 3.12, `pip install -r requirements.txt`, an `ANTHROPIC_API_KEY` in `.env`, Hugging Face access to the three models, and a 24 GB GPU. See `CLAUDE.md` for commands.

## Local evaluation

A fully local evaluation layer (no API calls; `LLM_OFFLINE=1` and API clients cannot be constructed) grades every
closed-book system: 3 base models, the 3 one-epoch fine-tunes of run_2026-10-03, and 3 models × 3 epochs.

```bash
tmux new -s localeval 'bash scripts/run_all_localeval.sh'   # resumable; markers in results/local_eval/.done/
tail -f logs/localeval.log
```

Stages: S1 judge calibration against the Opus grades (`src/local_judge.py calibrate`), S2 full grading
(`grade`), S3/S4 key-fact and claim decomposition (`src/local_facts.py`), S5 MiniCheck-7B / Flan-T5 / DeBERTa NLI
(`src/local_checks.py`), S6 statistics (`src/local_stats.py`), S7 paper pack (`src/local_pack.py`). Models run
through vLLM in `.venv-vllm` (`src/vllm_json_worker.py`). Outputs: `results/local_eval/` (calibration, metrics,
significance, per-item audit files, STATUS.md) and `results/paper_pack/` (figures, LaTeX tables, RESULTS.md).
