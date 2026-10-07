# Matching-base regeneration + research_materials: STATUS

_Written 2026-10-07 02:47. Start 2026-10-06 21:38:46; checkpoint 2026-10-07 00:33:46; hard limit 2026-10-07 03:38:46._

| Stage | Result | Minutes | Note |
|---|---|---|---|
| R1_dequant_qwen3-8b | done | 0 |  |
| R1_gen_qwen3-8b | done | 7 |  |
| R1_dequant_gemma-4-e4b-it | done | 0 |  |
| R1_gen_gemma-4-e4b-it | done | 15 |  |
| R1_dequant_llama-3.1-8b-instruct | done | 0 |  |
| R1_gen_llama-3.1-8b-instruct | done | 14 |  |
| R_grade_P12 | done | 50 |  |
| R_checks_P12 | done | 17 |  |
| R_stats_1 | done | 0 |  |
| R_materials_1 | done | 0 |  |
| R_grade_rest | done | 104 |  |
| R_checks_rest | done | 40 |  |
| deployment_variants_matching_base | pending | – | merged AWQ/RTN/GGUF on the dequantized base need the dequantized checkpoint + llm-compressor + a further judge/MiniCheck swap; resume: see STATUS.md |
| R_stats_final | done | 0 |  |
| R_materials_final | done | 0 |  |

## Dequantized-base verification (QA-only ep3 adapter, 20 training items, tolerance 2%)

- qwen3-8b: NF4 0.07177 vs dequantized 0.07189 (rel 0.171%) -> PASS
- gemma-4-e4b-it: NF4 0.05237 vs dequantized 0.05168 (rel 1.305%) -> PASS
- llama-3.1-8b-instruct: NF4 0.01154 vs dequantized 0.01137 (rel 1.491%) -> PASS

## Coverage per priority tier

- P1 (Qwen + Gemma QA-only ep1-3): 6 systems, 10,674 answers, 10,670 judge-graded, 10,674 with MiniCheck-7B checks
- P2 (Llama QA-only ep1-3): 3 systems, 5,337 answers, 5,337 judge-graded, 5,337 with MiniCheck-7B checks
- P3 (mixed-CPT adapters): 4 systems, 7,116 answers, 7,113 judge-graded, 7,116 with MiniCheck-7B checks
- P4 (base on NF4 weights (control)): 3 systems, 5,337 answers, 5,332 judge-graded, 5,337 with MiniCheck-7B checks
- D (1-epoch run, seed 43, Gemma base/concise re-render): 8 systems, 14,232 answers, 14,222 judge-graded, 14,232 with MiniCheck-7B checks

## Decisions, pending work and resume commands

- Disk: judge (15 GB) and MiniCheck-7B (15.5 GB) cannot coexist with a 10 GB reserve, so to have P1-P2 graded AND checked by the checkpoint the swap runs twice: judge -> MiniCheck for P1-P2, then judge -> MiniCheck again for the rest (two extra ~2 min downloads).
- Earlier results are untouched and relabelled '__bf16serve' in the reports (adapter on the bf16 base).
- PENDING deployment_variants_matching_base: merged AWQ/RTN/GGUF on the dequantized base need the dequantized checkpoint + llm-compressor + a further judge/MiniCheck swap; resume: see STATUS.md
- Resume anything unfinished: `bash scripts/resume_regen_after_crash.sh` (skips finished stages). Single steps: `python src/regen.py dequant|gen --model <m>`, `python src/regen.py grade|checks --tier P12|rest`, `python src/regen.py stats`, `python src/build_research_materials.py`.
- Deployment variants on the matching base (merged AWQ/RTN/GGUF): not run; they need the dequantized checkpoint, llm-compressor (src/trained_quant.py with --model models_dequant/<m>) and a further judge + MiniCheck swap.
- GPU since start: max 57 °C, max power draw 261 W.
- Disk free now: 13.3 GB.
- Failed push attempts: 0.
