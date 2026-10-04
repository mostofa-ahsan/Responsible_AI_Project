# 3-epoch QLoRA run: progress

_Updated 2026-10-03 22:45. Same recipe as run_2026-10-03, except 3 epochs (cosine over all 3). No API calls (LLM_OFFLINE=1); judge grading comes later._

## llama-3.1-8b-instruct: 2/3 epochs done

| Epoch | Train loss (last / mean) | Val loss | Minutes | Peak VRAM (GB) | test_indomain F1 / ROUGE-L | test_heldout_docs F1 / ROUGE-L | test_seen_facts F1 / ROUGE-L |
|---|---|---|---|---|---|---|---|
| 1 | 0.87813 / 1.5197 | 2.29331 | 68.7 | 10.3 | 0.357 / 0.270 | 0.331 / 0.246 | 0.363 / 0.281 |
| 2 | 0.10299 / 0.2862 | 3.18984 | 68.8 | 10.3 | 0.336 / 0.247 | 0.309 / 0.222 | 0.463 / 0.397 |

## qwen3-8b: 0/3 epochs done
_not started_

## gemma-4-e4b-it: 0/3 epochs done
_not started_

**Queue ETA:** 7 epoch(s) left, ~76 min each (training + 3 generations) -> about 07:37 (tomorrow).

Adapters: `models_epochs/<model>/epoch{1,2,3}/adapter`; logs: `models_epochs/<model>/{train_loss.csv,epochs.json}`; answers and metrics: `results/run_epochs/<model>/epoch<N>/`.
