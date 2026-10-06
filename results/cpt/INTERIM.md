# Arm C (mixed continued pretraining): interim results

_Updated 2026-10-06 14:10 by `src/cpt.py interim`. Judge grades come in Stage 4; '–' = not yet measured. Key-fact recall: MiniCheck-7B on the cached key facts. Exact repro. = ROUGE-L ≥ 0.8 vs the trained answer._

## Llama 3.1 8B (LoRA r=128, micro-batch 4)

| Epoch | minutes | train loss (raw / QA) | val QA loss | held-out ppl | peak VRAM (GB) |
|---|---|---|---|---|---|
| 0 (base) | – | – | 2.611 | 10.834 | – |
| 1 | 149.6 | 2.057 / 1.676 | 1.711 | 9.974 | 20.1 |
| 2 | 148.4 | 1.617 / 0.643 | 1.906 | 10.553 | 20.1 |

| Test | Variant | F1 | ROUGE-L | Words | Exact repro. | KF recall | Contra. |
|---|---|---|---|---|---|---|---|
| trained_exact | base | 0.179 | 0.129 | 222.1 | 0.000 | 0.517 | 0.100 |
| trained_exact | concise40 | 0.324 | 0.238 | 35.0 | 0.000 | 0.369 | 0.117 |
| trained_exact | ep1 | 0.481 | 0.413 | 35.1 | 0.080 | 0.430 | 0.058 |
| trained_exact | ep2 | 0.811 | 0.788 | 38.8 | 0.658 | 0.726 | 0.029 |
| trained_exact | ep3 | 0.856 | 0.839 | 38.1 | 0.742 | 0.751 | 0.019 |
| trained_exact | cpt_ep1 | 0.426 | 0.346 | 35.1 | 0.026 | 0.406 | 0.066 |
| trained_exact | cpt_ep2 | 0.619 | 0.565 | 38.0 | 0.308 | 0.583 | 0.033 |
| seen_facts | base | 0.154 | 0.107 | 227.9 | 0.000 | 0.498 | 0.083 |
| seen_facts | concise40 | 0.265 | 0.186 | 37.5 | 0.000 | 0.356 | 0.115 |
| seen_facts | ep1 | 0.363 | 0.281 | 38.0 | 0.025 | 0.387 | 0.078 |
| seen_facts | ep2 | 0.463 | 0.397 | 43.2 | 0.125 | 0.487 | 0.065 |
| seen_facts | ep3 | 0.473 | 0.411 | 42.2 | 0.143 | 0.485 | 0.058 |
| seen_facts | cpt_ep1 | 0.324 | 0.242 | 38.2 | 0.007 | 0.372 | 0.081 |
| seen_facts | cpt_ep2 | 0.398 | 0.320 | 40.8 | 0.075 | 0.419 | 0.081 |
| indomain | base | 0.182 | 0.130 | 219.7 | 0.000 | 0.555 | 0.091 |
| indomain | concise40 | 0.322 | 0.239 | 37.0 | 0.000 | 0.418 | 0.102 |
| indomain | ep1 | 0.358 | 0.269 | 36.0 | 0.002 | 0.372 | 0.087 |
| indomain | ep2 | 0.338 | 0.245 | 40.9 | 0.000 | 0.371 | 0.093 |
| indomain | ep3 | 0.334 | 0.243 | 40.3 | 0.000 | 0.371 | 0.077 |
| indomain | cpt_ep1 | 0.380 | 0.290 | 35.5 | 0.002 | 0.374 | 0.069 |
| indomain | cpt_ep2 | 0.352 | 0.262 | 38.2 | 0.000 | 0.373 | 0.085 |
| heldout_docs | base | 0.179 | 0.127 | 212.8 | 0.000 | 0.493 | 0.100 |
| heldout_docs | concise40 | 0.311 | 0.229 | 35.8 | 0.000 | 0.375 | 0.102 |
| heldout_docs | ep1 | 0.334 | 0.248 | 36.1 | 0.002 | 0.325 | 0.093 |
| heldout_docs | ep2 | 0.310 | 0.222 | 42.8 | 0.002 | 0.311 | 0.102 |
| heldout_docs | ep3 | 0.311 | 0.226 | 41.1 | 0.002 | 0.308 | 0.090 |
| heldout_docs | cpt_ep1 | 0.352 | 0.263 | 35.8 | 0.004 | 0.340 | 0.086 |
| heldout_docs | cpt_ep2 | 0.316 | 0.232 | 39.8 | 0.002 | 0.323 | 0.087 |

## Qwen3-8B (LoRA r=128, micro-batch 4)

| Epoch | minutes | train loss (raw / QA) | val QA loss | held-out ppl | peak VRAM (GB) |
|---|---|---|---|---|---|
| 0 (base) | – | – | 4.060 | 10.365 | – |
| 1 | 282.4 | 1.976 / 1.722 | 1.697 | 8.911 | 22.6 |

| Test | Variant | F1 | ROUGE-L | Words | Exact repro. | KF recall | Contra. |
|---|---|---|---|---|---|---|---|
| trained_exact | base | 0.266 | 0.186 | 123.9 | 0.000 | 0.559 | 0.093 |
| trained_exact | concise40 | 0.320 | 0.247 | 24.8 | 0.004 | 0.337 | 0.096 |
| trained_exact | ep1 | 0.445 | 0.368 | 32.5 | 0.054 | 0.399 | 0.069 |
| trained_exact | ep2 | 0.537 | 0.474 | 38.7 | 0.184 | 0.511 | 0.048 |
| trained_exact | ep3 | 0.561 | 0.502 | 38.1 | 0.214 | 0.529 | 0.055 |
| trained_exact | cpt_ep1 | 0.408 | 0.324 | 32.4 | 0.022 | 0.363 | 0.071 |
| seen_facts | base | 0.213 | 0.141 | 141.8 | 0.000 | 0.523 | 0.108 |
| seen_facts | concise40 | 0.263 | 0.191 | 25.1 | 0.000 | 0.316 | 0.105 |
| seen_facts | ep1 | 0.336 | 0.255 | 34.6 | 0.011 | 0.338 | 0.086 |
| seen_facts | ep2 | 0.366 | 0.283 | 42.3 | 0.025 | 0.426 | 0.083 |
| seen_facts | ep3 | 0.357 | 0.279 | 43.0 | 0.025 | 0.407 | 0.078 |
| seen_facts | cpt_ep1 | 0.311 | 0.231 | 34.5 | 0.004 | 0.333 | 0.082 |
| indomain | base | 0.265 | 0.181 | 124.2 | 0.000 | 0.564 | 0.096 |
| indomain | concise40 | 0.305 | 0.231 | 25.0 | 0.000 | 0.365 | 0.096 |
| indomain | ep1 | 0.363 | 0.275 | 32.9 | 0.002 | 0.360 | 0.070 |
| indomain | ep2 | 0.343 | 0.254 | 39.7 | 0.000 | 0.381 | 0.085 |
| indomain | ep3 | 0.343 | 0.253 | 40.1 | 0.000 | 0.385 | 0.079 |
| indomain | cpt_ep1 | 0.378 | 0.288 | 32.5 | 0.004 | 0.372 | 0.075 |
| heldout_docs | base | 0.259 | 0.175 | 121.7 | 0.000 | 0.518 | 0.088 |
| heldout_docs | concise40 | 0.291 | 0.219 | 25.0 | 0.004 | 0.347 | 0.097 |
| heldout_docs | ep1 | 0.340 | 0.256 | 33.2 | 0.004 | 0.345 | 0.101 |
| heldout_docs | ep2 | 0.319 | 0.233 | 40.7 | 0.002 | 0.323 | 0.085 |
| heldout_docs | ep3 | 0.315 | 0.227 | 39.9 | 0.002 | 0.337 | 0.091 |
| heldout_docs | cpt_ep1 | 0.355 | 0.274 | 32.8 | 0.004 | 0.323 | 0.087 |

## Gemma 4 E4B (LoRA r=64, micro-batch 1)

| Epoch | minutes | train loss (raw / QA) | val QA loss | held-out ppl | peak VRAM (GB) |
|---|---|---|---|---|---|
| 0 (base) | – | – | 4.357 | 46.472 | – |
| 1 | 242.3 | 2.145 / 1.875 | 1.751 | 10.247 | 23.1 |

| Test | Variant | F1 | ROUGE-L | Words | Exact repro. | KF recall | Contra. |
|---|---|---|---|---|---|---|---|
| trained_exact | base | 0.173 | 0.118 | 227.9 | 0.000 | 0.562 | 0.073 |
| trained_exact | concise40 | 0.289 | 0.214 | 26.4 | 0.002 | 0.316 | 0.082 |
| trained_exact | ep1 | 0.422 | 0.341 | 30.8 | 0.026 | 0.365 | 0.073 |
| trained_exact | ep2 | 0.467 | 0.394 | 33.6 | 0.066 | 0.422 | 0.068 |
| trained_exact | ep3 | 0.486 | 0.413 | 34.0 | 0.090 | 0.432 | 0.061 |
| trained_exact | cpt_ep1 | 0.394 | 0.312 | 29.7 | 0.010 | – | 0.076 |
| seen_facts | base | 0.151 | 0.101 | 237.5 | 0.000 | 0.530 | 0.063 |
| seen_facts | concise40 | 0.248 | 0.179 | 26.6 | 0.000 | 0.303 | 0.077 |
| seen_facts | ep1 | 0.340 | 0.256 | 34.3 | 0.014 | 0.359 | 0.085 |
| seen_facts | ep2 | 0.359 | 0.272 | 36.6 | 0.018 | 0.386 | 0.090 |
| seen_facts | ep3 | 0.368 | 0.279 | 37.6 | 0.018 | 0.402 | 0.086 |
| seen_facts | cpt_ep1 | 0.305 | 0.224 | 33.0 | 0.004 | – | 0.100 |
| indomain | base | 0.177 | 0.120 | 227.9 | 0.000 | 0.574 | 0.078 |
| indomain | concise40 | 0.287 | 0.211 | 26.6 | 0.000 | 0.367 | 0.067 |
| indomain | ep1 | 0.369 | 0.280 | 31.2 | 0.006 | 0.371 | 0.069 |
| indomain | ep2 | 0.351 | 0.263 | 34.9 | 0.000 | 0.364 | 0.070 |
| indomain | ep3 | 0.340 | 0.252 | 35.0 | 0.000 | 0.369 | 0.076 |
| indomain | cpt_ep1 | 0.377 | 0.292 | 31.2 | 0.008 | – | 0.073 |
| heldout_docs | base | 0.177 | 0.118 | 222.4 | 0.000 | 0.541 | 0.065 |
| heldout_docs | concise40 | 0.276 | 0.204 | 27.0 | 0.000 | 0.343 | 0.068 |
| heldout_docs | ep1 | 0.347 | 0.267 | 31.6 | 0.002 | 0.322 | 0.096 |
| heldout_docs | ep2 | 0.332 | 0.246 | 35.0 | 0.002 | 0.327 | 0.093 |
| heldout_docs | ep3 | 0.326 | 0.239 | 35.9 | 0.002 | 0.335 | 0.085 |
| heldout_docs | cpt_ep1 | 0.363 | 0.278 | 31.0 | 0.008 | – | 0.082 |

