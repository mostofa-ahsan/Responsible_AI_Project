**Table 11. Serving precision: adapter on the bf16 base vs on its training (NF4) base**

| Model | Adapter | Test | Metric | bf16 base | Training base | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B | mixed CPT ep1 | heldout_docs | exact_repro | 0.008 | 0.008 | +0.000 [-0.006, +0.006] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | heldout_docs | judge_lenient | 0.533 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | mixed CPT ep1 | heldout_docs | keyfact_recall | 0.335 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | mixed CPT ep1 | indomain | exact_repro | 0.008 | 0.008 | +0.000 [-0.006, +0.006] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | indomain | judge_lenient | 0.547 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | mixed CPT ep1 | indomain | keyfact_recall | 0.365 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | mixed CPT ep1 | seen_facts | exact_repro | 0.004 | 0.004 | +0.000 [+0.000, +0.000] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | seen_facts | judge_lenient | 0.554 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | mixed CPT ep1 | seen_facts | keyfact_recall | 0.317 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | mixed CPT ep1 | trained_exact | exact_repro | 0.010 | 0.012 | +0.002 [+0.000, +0.006] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | trained_exact | judge_lenient | 0.534 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | mixed CPT ep1 | trained_exact | keyfact_recall | 0.333 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only ep1 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | heldout_docs | judge_lenient | 0.500 | 0.493 | -0.007 [-0.032, +0.017] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | heldout_docs | keyfact_recall | 0.322 | 0.318 | -0.004 [-0.026, +0.018] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | indomain | exact_repro | 0.006 | 0.006 | +0.000 [+0.000, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | indomain | judge_lenient | 0.553 | 0.541 | -0.011 [-0.036, +0.014] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | indomain | keyfact_recall | 0.371 | 0.353 | -0.018 [-0.039, +0.004] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | seen_facts | exact_repro | 0.014 | 0.018 | +0.004 [-0.007, +0.018] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | seen_facts | judge_lenient | 0.595 | 0.593 | -0.002 [-0.038, +0.032] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | seen_facts | keyfact_recall | 0.359 | 0.366 | +0.007 [-0.024, +0.037] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | trained_exact | exact_repro | 0.026 | 0.042 | +0.016 [+0.006, +0.028] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | trained_exact | judge_lenient | 0.563 | 0.566 | +0.003 [-0.023, +0.029] | 1.0000 |
| Gemma 4 E4B | QA-only ep1 | trained_exact | keyfact_recall | 0.365 | 0.376 | +0.011 [-0.014, +0.037] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | heldout_docs | judge_lenient | 0.498 | 0.472 | -0.026 [-0.052, +0.002] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | heldout_docs | keyfact_recall | 0.327 | 0.304 | -0.023 [-0.045, -0.003] | 0.7031 |
| Gemma 4 E4B | QA-only ep2 | indomain | exact_repro | 0.000 | 0.002 | +0.002 [+0.000, +0.006] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | indomain | judge_lenient | 0.530 | 0.511 | -0.019 [-0.045, +0.006] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | indomain | keyfact_recall | 0.364 | 0.362 | -0.002 [-0.024, +0.020] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | seen_facts | exact_repro | 0.018 | 0.075 | +0.057 [+0.025, +0.090] | 0.1239 |
| Gemma 4 E4B | QA-only ep2 | seen_facts | judge_lenient | 0.599 | 0.622 | +0.023 [-0.014, +0.063] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | seen_facts | keyfact_recall | 0.386 | 0.414 | +0.028 [-0.007, +0.065] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | trained_exact | exact_repro | 0.066 | 0.418 | +0.352 [+0.310, +0.394] | 0.0365 |
| Gemma 4 E4B | QA-only ep2 | trained_exact | judge_lenient | 0.614 | 0.771 | +0.157 [+0.124, +0.190] | 0.0180 |
| Gemma 4 E4B | QA-only ep2 | trained_exact | keyfact_recall | 0.422 | 0.618 | +0.196 [+0.161, +0.230] | 0.0180 |
| Gemma 4 E4B | QA-only ep3 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | heldout_docs | judge_lenient | 0.484 | 0.449 | -0.035 [-0.063, -0.008] | 0.3123 |
| Gemma 4 E4B | QA-only ep3 | heldout_docs | keyfact_recall | 0.335 | 0.285 | -0.051 [-0.075, -0.027] | 0.0180 |
| Gemma 4 E4B | QA-only ep3 | indomain | exact_repro | 0.000 | 0.002 | +0.002 [+0.000, +0.006] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | indomain | judge_lenient | 0.532 | 0.496 | -0.036 [-0.065, -0.010] | 0.2989 |
| Gemma 4 E4B | QA-only ep3 | indomain | keyfact_recall | 0.369 | 0.355 | -0.014 [-0.037, +0.010] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | seen_facts | exact_repro | 0.018 | 0.075 | +0.057 [+0.029, +0.090] | 0.0630 |
| Gemma 4 E4B | QA-only ep3 | seen_facts | judge_lenient | 0.629 | 0.613 | -0.016 [-0.057, +0.029] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | seen_facts | keyfact_recall | 0.402 | 0.435 | +0.033 [-0.011, +0.076] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | trained_exact | exact_repro | 0.090 | 0.590 | +0.500 [+0.454, +0.544] | 0.0365 |
| Gemma 4 E4B | QA-only ep3 | trained_exact | judge_lenient | 0.638 | 0.842 | +0.204 [+0.170, +0.238] | 0.0180 |
| Gemma 4 E4B | QA-only ep3 | trained_exact | keyfact_recall | 0.432 | 0.688 | +0.256 [+0.222, +0.289] | 0.0180 |
| Gemma 4 E4B | QA-only 1-epoch run | heldout_docs | exact_repro | 0.004 | 0.006 | +0.002 [-0.004, +0.010] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | heldout_docs | judge_lenient | 0.518 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only 1-epoch run | heldout_docs | keyfact_recall | 0.357 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only 1-epoch run | indomain | exact_repro | 0.006 | 0.004 | -0.002 [-0.006, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | indomain | judge_lenient | 0.543 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only 1-epoch run | indomain | keyfact_recall | 0.380 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only 1-epoch run | seen_facts | exact_repro | 0.004 | 0.007 | +0.004 [+0.000, +0.011] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | seen_facts | judge_lenient | 0.584 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only 1-epoch run | seen_facts | keyfact_recall | 0.350 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only 1-epoch run | trained_exact | exact_repro | 0.014 | 0.028 | +0.014 [+0.002, +0.026] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | trained_exact | judge_lenient | 0.547 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only 1-epoch run | trained_exact | keyfact_recall | 0.349 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only ep1, seed 43 | heldout_docs | exact_repro | 0.004 | 0.002 | -0.002 [-0.006, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | heldout_docs | judge_lenient | 0.495 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only ep1, seed 43 | heldout_docs | keyfact_recall | 0.332 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only ep1, seed 43 | indomain | exact_repro | 0.002 | 0.004 | +0.002 [-0.004, +0.010] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | indomain | judge_lenient | 0.531 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only ep1, seed 43 | indomain | keyfact_recall | 0.376 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only ep1, seed 43 | seen_facts | exact_repro | 0.004 | 0.007 | +0.004 [+0.000, +0.011] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | seen_facts | judge_lenient | 0.572 | – | +nan [+nan, +nan] | – |
| Gemma 4 E4B | QA-only ep1, seed 43 | seen_facts | keyfact_recall | 0.360 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep1 | heldout_docs | exact_repro | 0.004 | 0.006 | +0.002 [+0.000, +0.008] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | heldout_docs | judge_lenient | 0.502 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep1 | heldout_docs | keyfact_recall | 0.340 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep1 | indomain | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | indomain | judge_lenient | 0.539 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep1 | indomain | keyfact_recall | 0.374 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep1 | seen_facts | exact_repro | 0.007 | 0.018 | +0.011 [+0.000, +0.025] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | seen_facts | judge_lenient | 0.572 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep1 | seen_facts | keyfact_recall | 0.372 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep1 | trained_exact | exact_repro | 0.026 | 0.040 | +0.014 [+0.002, +0.028] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | trained_exact | judge_lenient | 0.564 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep1 | trained_exact | keyfact_recall | 0.406 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep2 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | heldout_docs | judge_lenient | 0.469 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep2 | heldout_docs | keyfact_recall | 0.323 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep2 | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | indomain | judge_lenient | 0.528 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep2 | indomain | keyfact_recall | 0.373 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep2 | seen_facts | exact_repro | 0.075 | 0.122 | +0.047 [+0.014, +0.079] | 0.3963 |
| Llama 3.1 8B | mixed CPT ep2 | seen_facts | judge_lenient | 0.620 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep2 | seen_facts | keyfact_recall | 0.419 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep2 | trained_exact | exact_repro | 0.308 | 0.510 | +0.202 [+0.162, +0.242] | 0.0365 |
| Llama 3.1 8B | mixed CPT ep2 | trained_exact | judge_lenient | 0.735 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | mixed CPT ep2 | trained_exact | keyfact_recall | 0.583 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only ep1 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | heldout_docs | judge_lenient | 0.460 | 0.445 | -0.015 [-0.040, +0.010] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | heldout_docs | keyfact_recall | 0.325 | 0.313 | -0.011 [-0.033, +0.011] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | indomain | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | indomain | judge_lenient | 0.525 | 0.507 | -0.018 [-0.043, +0.006] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | indomain | keyfact_recall | 0.372 | 0.340 | -0.032 [-0.052, -0.011] | 0.1039 |
| Llama 3.1 8B | QA-only ep1 | seen_facts | exact_repro | 0.025 | 0.043 | +0.018 [+0.004, +0.036] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | seen_facts | judge_lenient | 0.615 | 0.609 | -0.005 [-0.043, +0.029] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | seen_facts | keyfact_recall | 0.387 | 0.407 | +0.019 [-0.012, +0.052] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | trained_exact | exact_repro | 0.080 | 0.136 | +0.056 [+0.034, +0.080] | 0.0365 |
| Llama 3.1 8B | QA-only ep1 | trained_exact | judge_lenient | 0.607 | 0.636 | +0.029 [+0.001, +0.056] | 0.9345 |
| Llama 3.1 8B | QA-only ep1 | trained_exact | keyfact_recall | 0.430 | 0.460 | +0.030 [+0.006, +0.055] | 0.2989 |
| Llama 3.1 8B | QA-only ep2 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | heldout_docs | judge_lenient | 0.465 | 0.452 | -0.013 [-0.039, +0.013] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | heldout_docs | keyfact_recall | 0.311 | 0.300 | -0.011 [-0.034, +0.011] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | indomain | exact_repro | 0.000 | 0.002 | +0.002 [+0.000, +0.006] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | indomain | judge_lenient | 0.496 | 0.501 | +0.005 [-0.024, +0.035] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | indomain | keyfact_recall | 0.371 | 0.350 | -0.021 [-0.048, +0.003] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | seen_facts | exact_repro | 0.125 | 0.265 | +0.140 [+0.097, +0.186] | 0.0365 |
| Llama 3.1 8B | QA-only ep2 | seen_facts | judge_lenient | 0.674 | 0.719 | +0.045 [+0.009, +0.081] | 0.4318 |
| Llama 3.1 8B | QA-only ep2 | seen_facts | keyfact_recall | 0.487 | 0.544 | +0.057 [+0.016, +0.097] | 0.1499 |
| Llama 3.1 8B | QA-only ep2 | trained_exact | exact_repro | 0.658 | 0.900 | +0.242 [+0.204, +0.284] | 0.0365 |
| Llama 3.1 8B | QA-only ep2 | trained_exact | judge_lenient | 0.853 | 0.963 | +0.110 [+0.087, +0.135] | 0.0180 |
| Llama 3.1 8B | QA-only ep2 | trained_exact | keyfact_recall | 0.726 | 0.823 | +0.097 [+0.070, +0.123] | 0.0180 |
| Llama 3.1 8B | QA-only ep3 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | heldout_docs | judge_lenient | 0.462 | 0.440 | -0.022 [-0.047, +0.003] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | heldout_docs | keyfact_recall | 0.308 | 0.311 | +0.003 [-0.019, +0.026] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | indomain | judge_lenient | 0.514 | 0.490 | -0.024 [-0.051, +0.003] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | indomain | keyfact_recall | 0.371 | 0.346 | -0.025 [-0.049, -0.000] | 0.8696 |
| Llama 3.1 8B | QA-only ep3 | seen_facts | exact_repro | 0.143 | 0.269 | +0.125 [+0.079, +0.172] | 0.0365 |
| Llama 3.1 8B | QA-only ep3 | seen_facts | judge_lenient | 0.654 | 0.719 | +0.065 [+0.025, +0.104] | 0.0180 |
| Llama 3.1 8B | QA-only ep3 | seen_facts | keyfact_recall | 0.485 | 0.555 | +0.071 [+0.031, +0.112] | 0.0290 |
| Llama 3.1 8B | QA-only ep3 | trained_exact | exact_repro | 0.742 | 0.954 | +0.212 [+0.176, +0.250] | 0.0365 |
| Llama 3.1 8B | QA-only ep3 | trained_exact | judge_lenient | 0.896 | 0.986 | +0.090 [+0.071, +0.112] | 0.0180 |
| Llama 3.1 8B | QA-only ep3 | trained_exact | keyfact_recall | 0.751 | 0.851 | +0.100 [+0.079, +0.122] | 0.0180 |
| Llama 3.1 8B | QA-only 1-epoch run | heldout_docs | exact_repro | 0.002 | 0.004 | +0.002 [-0.004, +0.010] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | heldout_docs | judge_lenient | 0.485 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only 1-epoch run | heldout_docs | keyfact_recall | 0.333 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only 1-epoch run | indomain | exact_repro | 0.004 | 0.004 | +0.000 [-0.006, +0.006] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | indomain | judge_lenient | 0.520 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only 1-epoch run | indomain | keyfact_recall | 0.375 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only 1-epoch run | seen_facts | exact_repro | 0.014 | 0.014 | +0.000 [-0.011, +0.011] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | seen_facts | judge_lenient | 0.595 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only 1-epoch run | seen_facts | keyfact_recall | 0.358 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only 1-epoch run | trained_exact | exact_repro | 0.040 | 0.054 | +0.014 [-0.002, +0.030] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | trained_exact | judge_lenient | 0.571 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only 1-epoch run | trained_exact | keyfact_recall | 0.378 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only ep1, seed 43 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | heldout_docs | judge_lenient | 0.474 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only ep1, seed 43 | heldout_docs | keyfact_recall | 0.315 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only ep1, seed 43 | indomain | exact_repro | 0.002 | 0.004 | +0.002 [+0.000, +0.006] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | indomain | judge_lenient | 0.521 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only ep1, seed 43 | indomain | keyfact_recall | 0.358 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only ep1, seed 43 | seen_facts | exact_repro | 0.029 | 0.043 | +0.014 [-0.007, +0.036] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | seen_facts | judge_lenient | 0.600 | – | +nan [+nan, +nan] | – |
| Llama 3.1 8B | QA-only ep1, seed 43 | seen_facts | keyfact_recall | 0.380 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | mixed CPT ep1 | heldout_docs | exact_repro | 0.004 | 0.006 | +0.002 [+0.000, +0.006] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | heldout_docs | judge_lenient | 0.499 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | mixed CPT ep1 | heldout_docs | keyfact_recall | 0.323 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | mixed CPT ep1 | indomain | exact_repro | 0.004 | 0.004 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | indomain | judge_lenient | 0.557 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | mixed CPT ep1 | indomain | keyfact_recall | 0.372 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | mixed CPT ep1 | seen_facts | exact_repro | 0.004 | 0.011 | +0.007 [+0.000, +0.018] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | seen_facts | judge_lenient | 0.575 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | mixed CPT ep1 | seen_facts | keyfact_recall | 0.333 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | mixed CPT ep1 | trained_exact | exact_repro | 0.022 | 0.032 | +0.010 [-0.002, +0.022] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | trained_exact | judge_lenient | 0.569 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | mixed CPT ep1 | trained_exact | keyfact_recall | 0.363 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only ep1 | heldout_docs | exact_repro | 0.004 | 0.004 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep1 | heldout_docs | judge_lenient | 0.506 | 0.486 | -0.020 [-0.043, +0.002] | 1.0000 |
| Qwen3-8B | QA-only ep1 | heldout_docs | keyfact_recall | 0.345 | 0.311 | -0.034 [-0.055, -0.014] | 0.0560 |
| Qwen3-8B | QA-only ep1 | indomain | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep1 | indomain | judge_lenient | 0.535 | 0.521 | -0.014 [-0.039, +0.012] | 1.0000 |
| Qwen3-8B | QA-only ep1 | indomain | keyfact_recall | 0.360 | 0.329 | -0.031 [-0.053, -0.008] | 0.2279 |
| Qwen3-8B | QA-only ep1 | seen_facts | exact_repro | 0.011 | 0.014 | +0.004 [+0.000, +0.011] | 1.0000 |
| Qwen3-8B | QA-only ep1 | seen_facts | judge_lenient | 0.573 | 0.599 | +0.025 [-0.013, +0.061] | 1.0000 |
| Qwen3-8B | QA-only ep1 | seen_facts | keyfact_recall | 0.338 | 0.338 | +0.000 [-0.032, +0.033] | 1.0000 |
| Qwen3-8B | QA-only ep1 | trained_exact | exact_repro | 0.054 | 0.068 | +0.014 [-0.002, +0.030] | 1.0000 |
| Qwen3-8B | QA-only ep1 | trained_exact | judge_lenient | 0.593 | 0.590 | -0.003 [-0.027, +0.022] | 1.0000 |
| Qwen3-8B | QA-only ep1 | trained_exact | keyfact_recall | 0.399 | 0.402 | +0.002 [-0.022, +0.026] | 1.0000 |
| Qwen3-8B | QA-only ep2 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep2 | heldout_docs | judge_lenient | 0.477 | 0.439 | -0.038 [-0.063, -0.013] | 0.1484 |
| Qwen3-8B | QA-only ep2 | heldout_docs | keyfact_recall | 0.323 | 0.300 | -0.024 [-0.045, -0.002] | 0.6927 |
| Qwen3-8B | QA-only ep2 | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep2 | indomain | judge_lenient | 0.508 | 0.495 | -0.013 [-0.041, +0.014] | 1.0000 |
| Qwen3-8B | QA-only ep2 | indomain | keyfact_recall | 0.381 | 0.361 | -0.020 [-0.043, +0.005] | 1.0000 |
| Qwen3-8B | QA-only ep2 | seen_facts | exact_repro | 0.025 | 0.061 | +0.036 [+0.011, +0.061] | 0.5097 |
| Qwen3-8B | QA-only ep2 | seen_facts | judge_lenient | 0.590 | 0.627 | +0.038 [+0.005, +0.070] | 0.6897 |
| Qwen3-8B | QA-only ep2 | seen_facts | keyfact_recall | 0.426 | 0.404 | -0.022 [-0.056, +0.013] | 1.0000 |
| Qwen3-8B | QA-only ep2 | trained_exact | exact_repro | 0.184 | 0.496 | +0.312 [+0.268, +0.356] | 0.0365 |
| Qwen3-8B | QA-only ep2 | trained_exact | judge_lenient | 0.667 | 0.761 | +0.094 [+0.063, +0.125] | 0.0180 |
| Qwen3-8B | QA-only ep2 | trained_exact | keyfact_recall | 0.511 | 0.640 | +0.129 [+0.098, +0.163] | 0.0180 |
| Qwen3-8B | QA-only ep3 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep3 | heldout_docs | judge_lenient | 0.476 | 0.431 | -0.044 [-0.070, -0.018] | 0.0580 |
| Qwen3-8B | QA-only ep3 | heldout_docs | keyfact_recall | 0.337 | 0.303 | -0.034 [-0.057, -0.012] | 0.0810 |
| Qwen3-8B | QA-only ep3 | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep3 | indomain | judge_lenient | 0.536 | 0.494 | -0.042 [-0.068, -0.014] | 0.0840 |
| Qwen3-8B | QA-only ep3 | indomain | keyfact_recall | 0.385 | 0.365 | -0.020 [-0.044, +0.004] | 1.0000 |
| Qwen3-8B | QA-only ep3 | seen_facts | exact_repro | 0.025 | 0.050 | +0.025 [+0.007, +0.047] | 1.0000 |
| Qwen3-8B | QA-only ep3 | seen_facts | judge_lenient | 0.584 | 0.619 | +0.032 [-0.007, +0.068] | 1.0000 |
| Qwen3-8B | QA-only ep3 | seen_facts | keyfact_recall | 0.407 | 0.402 | -0.004 [-0.040, +0.030] | 1.0000 |
| Qwen3-8B | QA-only ep3 | trained_exact | exact_repro | 0.214 | 0.610 | +0.396 [+0.350, +0.440] | 0.0365 |
| Qwen3-8B | QA-only ep3 | trained_exact | judge_lenient | 0.670 | 0.835 | +0.165 [+0.133, +0.195] | 0.0180 |
| Qwen3-8B | QA-only ep3 | trained_exact | keyfact_recall | 0.529 | 0.696 | +0.167 [+0.135, +0.201] | 0.0180 |
| Qwen3-8B | QA-only 1-epoch run | heldout_docs | exact_repro | 0.008 | 0.004 | -0.004 [-0.010, +0.000] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | heldout_docs | judge_lenient | 0.485 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only 1-epoch run | heldout_docs | keyfact_recall | 0.341 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only 1-epoch run | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | indomain | judge_lenient | 0.534 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only 1-epoch run | indomain | keyfact_recall | 0.387 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only 1-epoch run | seen_facts | exact_repro | 0.011 | 0.007 | -0.004 [-0.011, +0.000] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | seen_facts | judge_lenient | 0.561 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only 1-epoch run | seen_facts | keyfact_recall | 0.352 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only 1-epoch run | trained_exact | exact_repro | 0.016 | 0.022 | +0.006 [-0.002, +0.014] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | trained_exact | judge_lenient | 0.546 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only 1-epoch run | trained_exact | keyfact_recall | 0.361 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only ep1, seed 43 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | heldout_docs | judge_lenient | 0.471 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only ep1, seed 43 | heldout_docs | keyfact_recall | 0.320 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only ep1, seed 43 | indomain | exact_repro | 0.002 | 0.004 | +0.002 [+0.000, +0.006] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | indomain | judge_lenient | 0.553 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only ep1, seed 43 | indomain | keyfact_recall | 0.381 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only ep1, seed 43 | seen_facts | exact_repro | 0.007 | 0.025 | +0.018 [+0.004, +0.036] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | seen_facts | judge_lenient | 0.604 | – | +nan [+nan, +nan] | – |
| Qwen3-8B | QA-only ep1, seed 43 | seen_facts | keyfact_recall | 0.340 | – | +nan [+nan, +nan] | – |

Same adapters, same pre-tokenized greedy decoding except the base weights: bf16 (earlier results) vs the NF4-dequantized bf16 weights the adapter was trained on. Paired bootstrap on identical items.
