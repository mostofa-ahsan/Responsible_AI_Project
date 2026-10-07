# research_materials

_Built 2026-10-07 02:47 by `python src/build_research_materials.py` (rerunnable; every number is read from result files). Main numbers: adapters served on their training base (NF4-dequantized); earlier bf16-served results carry the suffix `__bf16serve`._

| File | Description | Paper | Sources |
|---|---|---|---|
| `00_dataset/dataset_stats.xlsx` | Dataset statistics | Data section, Table 1 | data/splits/*.jsonl, data/qa_pairs/full_v1*.jsonl, data/chunks/chunks.jsonl, data/inventory.csv, data/parsed/metadata.csv, logs/llm_usage.jsonl |
| `00_dataset/leakage_audit.json` | Leakage audit (corpus for mixed CPT; held-out document overlap) | Data section | /home/ahsan_linux/Responsible_AI_Project/data/cpt/leakage_audit.json |
| `00_dataset/qa_sample.xlsx` | QA pair samples | Appendix (data examples) | data/splits/*.jsonl |
| `01_training/training_runs.xlsx` | Training runs | Method section, Table 2 | config.yaml, models*/**/epochs.json, models/*/train_summary.json, results/*/stage_status.tsv |
| `01_training/model_footprint.csv` | Model footprint: base parameters, LoRA parameters, trainable %, adapter size | Method section | adapter_model.safetensors files, HF cache base checkpoints |
| `02_validation/judge_calibration.xlsx` | Judge calibration | Evaluation section, Table 4, Figure 7 | results/local_eval/calibration_A.json |
| `02_validation/checker_validation.xlsx` | Checker validation | Evaluation section, Table 4 | results/local_eval/validation.json |
| `02_validation/seed_replication.xlsx` | Seed replication | Limitations / robustness | results/seed_replication/summary.csv, judge_summary.csv |
| `03_results/main_results.xlsx` | Main results | Results section, Table 7, Figures 2-5 | results/regen/metrics_by_system.csv |
| `03_results/significance_tests.xlsx` | Significance tests | Results section, Table 7 markers | results/regen/significance.csv |
| `03_results/serving_precision.xlsx` | Adapter on the bf16 base vs on its training (NF4) base | Results section, serving-precision table, Figure 8 | results/regen/serving_precision.csv |
| `03_results/length_control.xlsx` | Base vs concise base vs best fine-tuned systems (length control) | Results section, Figure 5 | results/regen/length_control.csv |
| `03_results/robustness_gap.xlsx` | Exact minus reworded accuracy on the 279 paired items | Results section, Figure 4 | results/regen/robustness_gap.csv |
| `03_results/breakdown_qtype_dimension.xlsx` | Breakdown by question type and dimension | Appendix, Figure 9 | results/regen/master_items.csv.gz |
| `03_results/per_item_results.csv.gz` | Every answer with its judge grade, key-fact checks, F1/ROUGE-L, length (all systems) | Supplementary data | results/regen/master_items.csv.gz |
| `04_figures/F2_accuracy_four_tests.pdf` | Judge accuracy (correct + 0.5 partial) on the four test types for the base model, the concise base (≤ 40 words), the best QA-only epoch and the best mixed-CPT epoch (adapters served on their training base); 95% bootstrap CIs. | Figure 2 | see CAPTIONS.md |
| `04_figures/F3_accuracy_and_val_loss_vs_epoch.pdf` | Mean judge accuracy on unseen questions (in-domain + held-out) against epoch (0 = base; coloured, left axis) and validation loss (grey, right axis), QA-only (solid) and mixed CPT (dashed). | Figure 3 | see CAPTIONS.md |
| `04_figures/F4_exact_vs_reworded.pdf` | Exact training question vs its rewording on the 279 paired items (filled: exact, open: reworded) for base, best QA-only and best mixed-CPT systems. | Figure 4 | see CAPTIONS.md |
| `04_figures/F5_length_decomposition.pdf` | Held-out documents: change in judge accuracy from shortening the base answer (concise base − base) and from fine-tuning at matched length (best QA-only − concise base). | Figure 5 | see CAPTIONS.md |
| `04_figures/F6_metrics_vs_opus.pdf` | Token F1 (left) and key-fact recall (right) against Opus 5.5 accuracy for the 18 system × test cells graded by Opus (bf16-served base and 1-epoch systems); Pearson r shown. | Figure 6 | see CAPTIONS.md |
| `04_figures/F7_judge_validation.pdf` | Local judge vs Opus 5.5 on held-out calibration items: confusion matrix (left) and system × test accuracy (right; circles base, triangles 1-epoch fine-tuned). | Figure 7 | see CAPTIONS.md |
| `04_figures/F8_serving_precision_deployment.pdf` | Left: the best QA-only adapter served on the bf16 base (open) vs on its NF4 training base (filled). Right: deployment variants that exist (earlier deployment run; merged 4-bit variants were merged into the bf16 base). | Figure 8 | see CAPTIONS.md |
| `04_figures/F9_gain_heatmap.pdf` | Appendix: change in judge accuracy (best QA-only epoch on its training base − base) by question type × readiness dimension, unseen questions (in-domain + held-out), three models pooled. | Figure 9 | see CAPTIONS.md |
| `04_figures/CAPTIONS.md` | Figure captions | Figures | src/build_research_materials.py |
| `05_examples/T5_wins.xlsx` | Table 5: fine-tuning wins | Results section, Table 5 | results/regen/master_items.csv.gz |
| `05_examples/T6_side_by_side.xlsx` | Table 6: side-by-side answers | Results section, Table 6 | results/regen/master_items.csv.gz |
| `05_examples/SOURCES.md` | Bibliographic data of the source documents of the examples (missing fields flagged) | Results section, Tables 5-6 | results/regen/master_items.csv.gz, data/parsed/metadata.csv |
| `05_examples/sources.bib` | BibTeX of the source documents | Results section, Tables 5-6 | results/regen/master_items.csv.gz, data/parsed/metadata.csv |
| `05_examples/selection_log.json` | Example selection rules and picks | Results section, Tables 5-6 | results/regen/master_items.csv.gz, data/parsed/metadata.csv |
| `05_examples/T5_wins.md` | Table 5 (Markdown) | Results section, Tables 5-6 | results/regen/master_items.csv.gz, data/parsed/metadata.csv |
| `05_examples/T5_wins.tex` | Table 5 (LaTeX) | Results section, Tables 5-6 | results/regen/master_items.csv.gz, data/parsed/metadata.csv |
| `05_examples/T6_side_by_side.md` | Table 6 (Markdown) | Results section, Tables 5-6 | results/regen/master_items.csv.gz, data/parsed/metadata.csv |
| `05_examples/T6_side_by_side.tex` | Table 6 (LaTeX) | Results section, Tables 5-6 | results/regen/master_items.csv.gz, data/parsed/metadata.csv |
| `06_diagnostics/format_check.md` | Prompt-format check / dequantized-base verification | Methods (serving precision), appendix | results/diagnostics/format_check.md |
| `06_diagnostics/dequant_check.json` | Prompt-format check / dequantized-base verification | Methods (serving precision), appendix | results/diagnostics/dequant_check.json |
| `06_diagnostics/dequant_verify_gemma-4-e4b-it.json` | Prompt-format check / dequantized-base verification | Methods (serving precision), appendix | results/regen/dequant_verify_gemma-4-e4b-it.json |
| `06_diagnostics/dequant_verify_llama-3.1-8b-instruct.json` | Prompt-format check / dequantized-base verification | Methods (serving precision), appendix | results/regen/dequant_verify_llama-3.1-8b-instruct.json |
| `06_diagnostics/dequant_verify_qwen3-8b.json` | Prompt-format check / dequantized-base verification | Methods (serving precision), appendix | results/regen/dequant_verify_qwen3-8b.json |
| `07_deployment/deployment.xlsx` | Deployment variants | Discussion / deployment, Figure 8 | results/trained_eval/deployment.csv, deploy_stats.json |

## Gaps

- none
