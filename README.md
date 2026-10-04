# Responsible AI in Higher Education: QA dataset & fine-tuning experiments

Pipeline: PDF parsing (Docling) → structure-aware chunking → QA generation with Claude (Sonnet) → judge filters (Opus) → document-level splits → QLoRA fine-tuning of Qwen3-8B, Gemma 4 E4B and Llama 3.1 8B → closed-book evaluation (Opus judge, token F1, ROUGE-L).

- Dataset: 9,350 QA pairs from 100 books/articles (`docs/DATASET_CARD_full_v1.md`)
- Results: `results/run_2026-10-03/SUMMARY.md`
- Headline: 1-epoch closed-book QLoRA raises F1/ROUGE-L but lowers judge-rated accuracy on all three models.

Data, model weights and adapters are not included (source material is copyrighted).
To reproduce: Python 3.12, `pip install -r requirements.txt`, an `ANTHROPIC_API_KEY` in `.env`, Hugging Face access to the three models, and a 24 GB GPU. See `CLAUDE.md` for commands.
