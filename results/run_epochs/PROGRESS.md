# 3-epoch QLoRA run: progress

_Updated 2026-10-04 07:12. Same recipe as run_2026-10-03, except 3 epochs (cosine over all 3). No API calls (LLM_OFFLINE=1); judge grading comes later._

## llama-3.1-8b-instruct: 3/3 epochs done

| Epoch | Train loss (last / mean) | Val loss | Minutes | Peak VRAM (GB) | test_indomain F1 / ROUGE-L | test_heldout_docs F1 / ROUGE-L | test_seen_facts F1 / ROUGE-L |
|---|---|---|---|---|---|---|---|
| 1 | 0.87813 / 1.5197 | 2.29331 | 68.7 | 10.3 | 0.357 / 0.270 | 0.331 / 0.246 | 0.363 / 0.281 |
| 2 | 0.10299 / 0.2862 | 3.18984 | 68.8 | 10.3 | 0.336 / 0.247 | 0.309 / 0.222 | 0.463 / 0.397 |
| 3 | 0.02685 / 0.0232 | 3.5818 | 68.9 | 10.3 | 0.332 / 0.244 | 0.308 / 0.224 | 0.473 / 0.411 |

## qwen3-8b: 3/3 epochs done

| Epoch | Train loss (last / mean) | Val loss | Minutes | Peak VRAM (GB) | test_indomain F1 / ROUGE-L | test_heldout_docs F1 / ROUGE-L | test_seen_facts F1 / ROUGE-L |
|---|---|---|---|---|---|---|---|
| 1 | 1.12164 / 1.5971 | 2.09474 | 67.6 | 11.4 | 0.361 / 0.275 | 0.337 / 0.254 | 0.336 / 0.255 |
| 2 | 0.29647 / 0.5001 | 2.8929 | 67.7 | 11.4 | 0.341 / 0.253 | 0.316 / 0.228 | 0.366 / 0.283 |
| 3 | 0.11101 / 0.113 | 3.23212 | 67.7 | 11.4 | 0.339 / 0.249 | 0.312 / 0.226 | 0.357 / 0.279 |

## gemma-4-e4b-it: 3/3 epochs done

| Epoch | Train loss (last / mean) | Val loss | Minutes | Peak VRAM (GB) | test_indomain F1 / ROUGE-L | test_heldout_docs F1 / ROUGE-L | test_seen_facts F1 / ROUGE-L |
|---|---|---|---|---|---|---|---|
| 1 | 1.19796 / 1.6833 | 2.2419 | 65.2 | 19.8 | 0.372 / 0.285 | 0.344 / 0.262 | 0.340 / 0.256 |
| 2 | 0.31713 / 0.5781 | 4.02746 | 64.5 | 19.8 | 0.353 / 0.265 | 0.327 / 0.243 | 0.359 / 0.272 |
| 3 | 0.10079 / 0.1122 | 5.18349 | 64.9 | 19.8 | 0.344 / 0.256 | 0.323 / 0.238 | 0.368 / 0.279 |

**Queue ETA:** 0 epoch(s) left, ~75 min each (training + 3 generations) -> about 07:12.

Adapters: `models_epochs/<model>/epoch{1,2,3}/adapter`; logs: `models_epochs/<model>/{train_loss.csv,epochs.json}`; answers and metrics: `results/run_epochs/<model>/epoch<N>/`.

## Local evaluation

_Added 2026-10-04 22:51 by src/local_pack.py; local judge mistral-small-3.2-24b-awq; details in results/local_eval/SUMMARY.md and results/paper_pack/._

| Model | Variant | test_indomain judge acc / KF recall | test_heldout_docs judge acc / KF recall | test_seen_facts judge acc / KF recall |
|---|---|---|---|---|
| qwen3-8b | base | 0.669 / 0.564 | 0.642 / 0.518 | 0.649 / 0.523 |
| qwen3-8b | 1-epoch (run 10-03) | 0.534 / 0.387 | 0.485 / 0.341 | 0.561 / 0.352 |
| qwen3-8b | ep1 | 0.535 / 0.360 | 0.506 / 0.345 | 0.573 / 0.338 |
| qwen3-8b | ep2 | 0.508 / 0.381 | 0.477 / 0.323 | 0.590 / 0.426 |
| qwen3-8b | ep3 | 0.536 / 0.385 | 0.476 / 0.337 | 0.584 / 0.407 |
| gemma-4-e4b-it | base | 0.731 / 0.574 | 0.679 / 0.541 | 0.668 / 0.530 |
| gemma-4-e4b-it | 1-epoch (run 10-03) | 0.543 / 0.380 | 0.518 / 0.357 | 0.584 / 0.350 |
| gemma-4-e4b-it | ep1 | 0.553 / 0.371 | 0.500 / 0.322 | 0.595 / 0.359 |
| gemma-4-e4b-it | ep2 | 0.530 / 0.364 | 0.498 / 0.327 | 0.599 / 0.386 |
| gemma-4-e4b-it | ep3 | 0.532 / 0.369 | 0.484 / 0.335 | 0.629 / 0.402 |
| llama-3.1-8b-instruct | base | 0.575 / 0.555 | 0.534 / 0.493 | 0.522 / 0.498 |
| llama-3.1-8b-instruct | 1-epoch (run 10-03) | 0.520 / 0.375 | 0.485 / 0.333 | 0.595 / 0.358 |
| llama-3.1-8b-instruct | ep1 | 0.525 / 0.372 | 0.460 / 0.325 | 0.615 / 0.387 |
| llama-3.1-8b-instruct | ep2 | 0.496 / 0.371 | 0.465 / 0.311 | 0.674 / 0.487 |
| llama-3.1-8b-instruct | ep3 | 0.514 / 0.371 | 0.462 / 0.308 | 0.654 / 0.485 |
