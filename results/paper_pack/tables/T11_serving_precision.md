**Table 11. Serving precision: adapter on the bf16 base vs on its training (NF4) base**

| Model | Adapter | Test | Metric | bf16 base | Training base | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B | mixed CPT ep1 | heldout_docs | exact_repro | 0.008 | 0.008 | +0.000 [-0.006, +0.006] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | heldout_docs | judge_lenient | 0.533 | 0.496 | -0.037 [-0.060, -0.012] | 0.2519 |
| Gemma 4 E4B | mixed CPT ep1 | heldout_docs | keyfact_recall | 0.335 | 0.325 | -0.010 [-0.029, +0.009] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | indomain | exact_repro | 0.008 | 0.008 | +0.000 [-0.006, +0.006] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | indomain | judge_lenient | 0.547 | 0.527 | -0.020 [-0.045, +0.005] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | indomain | keyfact_recall | 0.365 | 0.371 | +0.006 [-0.014, +0.025] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | seen_facts | exact_repro | 0.004 | 0.004 | +0.000 [+0.000, +0.000] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | seen_facts | judge_lenient | 0.554 | 0.547 | -0.007 [-0.039, +0.023] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | seen_facts | keyfact_recall | 0.317 | 0.314 | -0.003 [-0.030, +0.025] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | trained_exact | exact_repro | 0.010 | 0.012 | +0.002 [+0.000, +0.006] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | trained_exact | judge_lenient | 0.534 | 0.516 | -0.018 [-0.041, +0.006] | 1.0000 |
| Gemma 4 E4B | mixed CPT ep1 | trained_exact | keyfact_recall | 0.333 | 0.330 | -0.003 [-0.024, +0.017] | 1.0000 |
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
| Gemma 4 E4B | QA-only ep2 | heldout_docs | keyfact_recall | 0.327 | 0.304 | -0.023 [-0.045, -0.003] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | indomain | exact_repro | 0.000 | 0.002 | +0.002 [+0.000, +0.006] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | indomain | judge_lenient | 0.530 | 0.511 | -0.019 [-0.045, +0.006] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | indomain | keyfact_recall | 0.364 | 0.362 | -0.002 [-0.024, +0.020] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | seen_facts | exact_repro | 0.018 | 0.075 | +0.057 [+0.025, +0.090] | 0.1239 |
| Gemma 4 E4B | QA-only ep2 | seen_facts | judge_lenient | 0.599 | 0.622 | +0.023 [-0.014, +0.063] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | seen_facts | keyfact_recall | 0.386 | 0.414 | +0.028 [-0.007, +0.065] | 1.0000 |
| Gemma 4 E4B | QA-only ep2 | trained_exact | exact_repro | 0.066 | 0.418 | +0.352 [+0.310, +0.394] | 0.0365 |
| Gemma 4 E4B | QA-only ep2 | trained_exact | judge_lenient | 0.614 | 0.771 | +0.157 [+0.124, +0.190] | 0.0365 |
| Gemma 4 E4B | QA-only ep2 | trained_exact | keyfact_recall | 0.422 | 0.618 | +0.196 [+0.161, +0.230] | 0.0365 |
| Gemma 4 E4B | QA-only ep3 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | heldout_docs | judge_lenient | 0.484 | 0.449 | -0.035 [-0.063, -0.008] | 0.7371 |
| Gemma 4 E4B | QA-only ep3 | heldout_docs | keyfact_recall | 0.335 | 0.285 | -0.051 [-0.075, -0.027] | 0.0365 |
| Gemma 4 E4B | QA-only ep3 | indomain | exact_repro | 0.000 | 0.002 | +0.002 [+0.000, +0.006] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | indomain | judge_lenient | 0.532 | 0.496 | -0.036 [-0.065, -0.010] | 0.6897 |
| Gemma 4 E4B | QA-only ep3 | indomain | keyfact_recall | 0.369 | 0.355 | -0.014 [-0.037, +0.010] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | seen_facts | exact_repro | 0.018 | 0.075 | +0.057 [+0.029, +0.090] | 0.0630 |
| Gemma 4 E4B | QA-only ep3 | seen_facts | judge_lenient | 0.629 | 0.613 | -0.016 [-0.057, +0.029] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | seen_facts | keyfact_recall | 0.402 | 0.435 | +0.033 [-0.011, +0.076] | 1.0000 |
| Gemma 4 E4B | QA-only ep3 | trained_exact | exact_repro | 0.090 | 0.590 | +0.500 [+0.454, +0.544] | 0.0365 |
| Gemma 4 E4B | QA-only ep3 | trained_exact | judge_lenient | 0.638 | 0.842 | +0.204 [+0.170, +0.238] | 0.0365 |
| Gemma 4 E4B | QA-only ep3 | trained_exact | keyfact_recall | 0.432 | 0.688 | +0.256 [+0.222, +0.289] | 0.0365 |
| Gemma 4 E4B | QA-only 1-epoch run | heldout_docs | exact_repro | 0.004 | 0.006 | +0.002 [-0.004, +0.010] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | heldout_docs | judge_lenient | 0.518 | 0.512 | -0.006 [-0.030, +0.017] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | heldout_docs | keyfact_recall | 0.357 | 0.332 | -0.025 [-0.043, -0.006] | 0.4718 |
| Gemma 4 E4B | QA-only 1-epoch run | indomain | exact_repro | 0.006 | 0.004 | -0.002 [-0.006, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | indomain | judge_lenient | 0.543 | 0.524 | -0.019 [-0.043, +0.005] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | indomain | keyfact_recall | 0.380 | 0.367 | -0.013 [-0.032, +0.007] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | seen_facts | exact_repro | 0.004 | 0.007 | +0.004 [+0.000, +0.011] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | seen_facts | judge_lenient | 0.584 | 0.556 | -0.029 [-0.063, +0.004] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | seen_facts | keyfact_recall | 0.350 | 0.337 | -0.013 [-0.043, +0.015] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | trained_exact | exact_repro | 0.014 | 0.028 | +0.014 [+0.002, +0.026] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | trained_exact | judge_lenient | 0.547 | 0.549 | +0.002 [-0.023, +0.027] | 1.0000 |
| Gemma 4 E4B | QA-only 1-epoch run | trained_exact | keyfact_recall | 0.349 | 0.355 | +0.006 [-0.017, +0.029] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | heldout_docs | exact_repro | 0.004 | 0.002 | -0.002 [-0.006, +0.000] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | heldout_docs | judge_lenient | 0.495 | 0.480 | -0.015 [-0.041, +0.011] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | heldout_docs | keyfact_recall | 0.332 | 0.314 | -0.018 [-0.037, +0.003] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | indomain | exact_repro | 0.002 | 0.004 | +0.002 [-0.004, +0.010] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | indomain | judge_lenient | 0.531 | 0.519 | -0.012 [-0.036, +0.012] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | indomain | keyfact_recall | 0.376 | 0.357 | -0.019 [-0.040, +0.002] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | seen_facts | exact_repro | 0.004 | 0.007 | +0.004 [+0.000, +0.011] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | seen_facts | judge_lenient | 0.572 | 0.557 | -0.014 [-0.048, +0.020] | 1.0000 |
| Gemma 4 E4B | QA-only ep1, seed 43 | seen_facts | keyfact_recall | 0.360 | 0.352 | -0.008 [-0.042, +0.022] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | heldout_docs | exact_repro | 0.004 | 0.006 | +0.002 [+0.000, +0.008] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | heldout_docs | judge_lenient | 0.502 | 0.471 | -0.031 [-0.055, -0.007] | 0.7536 |
| Llama 3.1 8B | mixed CPT ep1 | heldout_docs | keyfact_recall | 0.340 | 0.314 | -0.025 [-0.043, -0.006] | 0.4928 |
| Llama 3.1 8B | mixed CPT ep1 | indomain | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | indomain | judge_lenient | 0.539 | 0.537 | -0.002 [-0.027, +0.021] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | indomain | keyfact_recall | 0.374 | 0.367 | -0.007 [-0.027, +0.012] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | seen_facts | exact_repro | 0.007 | 0.018 | +0.011 [+0.000, +0.025] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | seen_facts | judge_lenient | 0.572 | 0.557 | -0.014 [-0.048, +0.018] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | seen_facts | keyfact_recall | 0.372 | 0.369 | -0.003 [-0.031, +0.027] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | trained_exact | exact_repro | 0.026 | 0.040 | +0.014 [+0.002, +0.028] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | trained_exact | judge_lenient | 0.564 | 0.574 | +0.010 [-0.015, +0.036] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep1 | trained_exact | keyfact_recall | 0.406 | 0.418 | +0.012 [-0.010, +0.037] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | heldout_docs | judge_lenient | 0.469 | 0.470 | +0.002 [-0.023, +0.028] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | heldout_docs | keyfact_recall | 0.323 | 0.307 | -0.016 [-0.039, +0.007] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | indomain | judge_lenient | 0.528 | 0.526 | -0.002 [-0.028, +0.025] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | indomain | keyfact_recall | 0.373 | 0.365 | -0.008 [-0.034, +0.017] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | seen_facts | exact_repro | 0.075 | 0.122 | +0.047 [+0.014, +0.079] | 0.3963 |
| Llama 3.1 8B | mixed CPT ep2 | seen_facts | judge_lenient | 0.620 | 0.667 | +0.047 [+0.014, +0.082] | 0.5792 |
| Llama 3.1 8B | mixed CPT ep2 | seen_facts | keyfact_recall | 0.419 | 0.444 | +0.025 [-0.007, +0.056] | 1.0000 |
| Llama 3.1 8B | mixed CPT ep2 | trained_exact | exact_repro | 0.308 | 0.510 | +0.202 [+0.162, +0.242] | 0.0365 |
| Llama 3.1 8B | mixed CPT ep2 | trained_exact | judge_lenient | 0.735 | 0.809 | +0.074 [+0.046, +0.101] | 0.0365 |
| Llama 3.1 8B | mixed CPT ep2 | trained_exact | keyfact_recall | 0.583 | 0.669 | +0.086 [+0.058, +0.113] | 0.0365 |
| Llama 3.1 8B | QA-only ep1 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | heldout_docs | judge_lenient | 0.460 | 0.445 | -0.015 [-0.040, +0.010] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | heldout_docs | keyfact_recall | 0.325 | 0.313 | -0.011 [-0.033, +0.011] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | indomain | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | indomain | judge_lenient | 0.525 | 0.507 | -0.018 [-0.043, +0.006] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | indomain | keyfact_recall | 0.372 | 0.340 | -0.032 [-0.052, -0.011] | 0.2479 |
| Llama 3.1 8B | QA-only ep1 | seen_facts | exact_repro | 0.025 | 0.043 | +0.018 [+0.004, +0.036] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | seen_facts | judge_lenient | 0.615 | 0.609 | -0.005 [-0.043, +0.029] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | seen_facts | keyfact_recall | 0.387 | 0.407 | +0.019 [-0.012, +0.052] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | trained_exact | exact_repro | 0.080 | 0.136 | +0.056 [+0.034, +0.080] | 0.0365 |
| Llama 3.1 8B | QA-only ep1 | trained_exact | judge_lenient | 0.607 | 0.636 | +0.029 [+0.001, +0.056] | 1.0000 |
| Llama 3.1 8B | QA-only ep1 | trained_exact | keyfact_recall | 0.430 | 0.460 | +0.030 [+0.006, +0.055] | 0.7276 |
| Llama 3.1 8B | QA-only ep2 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | heldout_docs | judge_lenient | 0.465 | 0.452 | -0.013 [-0.039, +0.013] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | heldout_docs | keyfact_recall | 0.311 | 0.300 | -0.011 [-0.034, +0.011] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | indomain | exact_repro | 0.000 | 0.002 | +0.002 [+0.000, +0.006] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | indomain | judge_lenient | 0.496 | 0.501 | +0.005 [-0.024, +0.035] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | indomain | keyfact_recall | 0.371 | 0.350 | -0.021 [-0.048, +0.003] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | seen_facts | exact_repro | 0.125 | 0.265 | +0.140 [+0.097, +0.186] | 0.0365 |
| Llama 3.1 8B | QA-only ep2 | seen_facts | judge_lenient | 0.674 | 0.719 | +0.045 [+0.009, +0.081] | 1.0000 |
| Llama 3.1 8B | QA-only ep2 | seen_facts | keyfact_recall | 0.487 | 0.544 | +0.057 [+0.016, +0.097] | 0.3658 |
| Llama 3.1 8B | QA-only ep2 | trained_exact | exact_repro | 0.658 | 0.900 | +0.242 [+0.204, +0.284] | 0.0365 |
| Llama 3.1 8B | QA-only ep2 | trained_exact | judge_lenient | 0.853 | 0.963 | +0.110 [+0.087, +0.135] | 0.0365 |
| Llama 3.1 8B | QA-only ep2 | trained_exact | keyfact_recall | 0.726 | 0.823 | +0.097 [+0.070, +0.123] | 0.0365 |
| Llama 3.1 8B | QA-only ep3 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | heldout_docs | judge_lenient | 0.462 | 0.440 | -0.022 [-0.047, +0.003] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | heldout_docs | keyfact_recall | 0.308 | 0.311 | +0.003 [-0.019, +0.026] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | indomain | judge_lenient | 0.514 | 0.490 | -0.024 [-0.051, +0.003] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | indomain | keyfact_recall | 0.371 | 0.346 | -0.025 [-0.049, -0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep3 | seen_facts | exact_repro | 0.143 | 0.269 | +0.125 [+0.079, +0.172] | 0.0365 |
| Llama 3.1 8B | QA-only ep3 | seen_facts | judge_lenient | 0.654 | 0.719 | +0.065 [+0.025, +0.104] | 0.0365 |
| Llama 3.1 8B | QA-only ep3 | seen_facts | keyfact_recall | 0.485 | 0.555 | +0.071 [+0.031, +0.112] | 0.0650 |
| Llama 3.1 8B | QA-only ep3 | trained_exact | exact_repro | 0.742 | 0.954 | +0.212 [+0.176, +0.250] | 0.0365 |
| Llama 3.1 8B | QA-only ep3 | trained_exact | judge_lenient | 0.896 | 0.986 | +0.090 [+0.071, +0.112] | 0.0365 |
| Llama 3.1 8B | QA-only ep3 | trained_exact | keyfact_recall | 0.751 | 0.851 | +0.100 [+0.079, +0.122] | 0.0365 |
| Llama 3.1 8B | QA-only 1-epoch run | heldout_docs | exact_repro | 0.002 | 0.004 | +0.002 [-0.004, +0.010] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | heldout_docs | judge_lenient | 0.485 | 0.474 | -0.011 [-0.036, +0.013] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | heldout_docs | keyfact_recall | 0.333 | 0.335 | +0.002 [-0.020, +0.023] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | indomain | exact_repro | 0.004 | 0.004 | +0.000 [-0.006, +0.006] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | indomain | judge_lenient | 0.520 | 0.527 | +0.008 [-0.016, +0.031] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | indomain | keyfact_recall | 0.375 | 0.345 | -0.030 [-0.051, -0.008] | 0.4198 |
| Llama 3.1 8B | QA-only 1-epoch run | seen_facts | exact_repro | 0.014 | 0.014 | +0.000 [-0.011, +0.011] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | seen_facts | judge_lenient | 0.595 | 0.588 | -0.005 [-0.040, +0.029] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | seen_facts | keyfact_recall | 0.358 | 0.365 | +0.007 [-0.020, +0.033] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | trained_exact | exact_repro | 0.040 | 0.054 | +0.014 [-0.002, +0.030] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | trained_exact | judge_lenient | 0.571 | 0.564 | -0.007 [-0.034, +0.017] | 1.0000 |
| Llama 3.1 8B | QA-only 1-epoch run | trained_exact | keyfact_recall | 0.378 | 0.383 | +0.005 [-0.017, +0.028] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | heldout_docs | judge_lenient | 0.474 | 0.454 | -0.020 [-0.046, +0.007] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | heldout_docs | keyfact_recall | 0.315 | 0.292 | -0.023 [-0.043, -0.003] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | indomain | exact_repro | 0.002 | 0.004 | +0.002 [+0.000, +0.006] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | indomain | judge_lenient | 0.521 | 0.517 | -0.004 [-0.028, +0.019] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | indomain | keyfact_recall | 0.358 | 0.361 | +0.003 [-0.023, +0.029] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | seen_facts | exact_repro | 0.029 | 0.043 | +0.014 [-0.007, +0.036] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | seen_facts | judge_lenient | 0.600 | 0.615 | +0.014 [-0.020, +0.048] | 1.0000 |
| Llama 3.1 8B | QA-only ep1, seed 43 | seen_facts | keyfact_recall | 0.380 | 0.403 | +0.023 [-0.011, +0.056] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | heldout_docs | exact_repro | 0.004 | 0.006 | +0.002 [+0.000, +0.006] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | heldout_docs | judge_lenient | 0.499 | 0.482 | -0.017 [-0.039, +0.008] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | heldout_docs | keyfact_recall | 0.323 | 0.326 | +0.003 [-0.016, +0.022] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | indomain | exact_repro | 0.004 | 0.004 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | indomain | judge_lenient | 0.557 | 0.544 | -0.013 [-0.037, +0.011] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | indomain | keyfact_recall | 0.372 | 0.377 | +0.005 [-0.014, +0.024] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | seen_facts | exact_repro | 0.004 | 0.011 | +0.007 [+0.000, +0.018] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | seen_facts | judge_lenient | 0.575 | 0.566 | -0.009 [-0.043, +0.025] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | seen_facts | keyfact_recall | 0.333 | 0.331 | -0.002 [-0.031, +0.024] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | trained_exact | exact_repro | 0.022 | 0.032 | +0.010 [-0.002, +0.022] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | trained_exact | judge_lenient | 0.569 | 0.568 | -0.002 [-0.026, +0.023] | 1.0000 |
| Qwen3-8B | mixed CPT ep1 | trained_exact | keyfact_recall | 0.363 | 0.358 | -0.005 [-0.024, +0.015] | 1.0000 |
| Qwen3-8B | QA-only ep1 | heldout_docs | exact_repro | 0.004 | 0.004 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep1 | heldout_docs | judge_lenient | 0.506 | 0.486 | -0.020 [-0.043, +0.002] | 1.0000 |
| Qwen3-8B | QA-only ep1 | heldout_docs | keyfact_recall | 0.345 | 0.311 | -0.034 [-0.055, -0.014] | 0.1279 |
| Qwen3-8B | QA-only ep1 | indomain | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep1 | indomain | judge_lenient | 0.535 | 0.521 | -0.014 [-0.039, +0.012] | 1.0000 |
| Qwen3-8B | QA-only ep1 | indomain | keyfact_recall | 0.360 | 0.329 | -0.031 [-0.053, -0.008] | 0.5412 |
| Qwen3-8B | QA-only ep1 | seen_facts | exact_repro | 0.011 | 0.014 | +0.004 [+0.000, +0.011] | 1.0000 |
| Qwen3-8B | QA-only ep1 | seen_facts | judge_lenient | 0.573 | 0.599 | +0.025 [-0.013, +0.061] | 1.0000 |
| Qwen3-8B | QA-only ep1 | seen_facts | keyfact_recall | 0.338 | 0.338 | +0.000 [-0.032, +0.033] | 1.0000 |
| Qwen3-8B | QA-only ep1 | trained_exact | exact_repro | 0.054 | 0.068 | +0.014 [-0.002, +0.030] | 1.0000 |
| Qwen3-8B | QA-only ep1 | trained_exact | judge_lenient | 0.593 | 0.590 | -0.003 [-0.027, +0.022] | 1.0000 |
| Qwen3-8B | QA-only ep1 | trained_exact | keyfact_recall | 0.399 | 0.402 | +0.002 [-0.022, +0.026] | 1.0000 |
| Qwen3-8B | QA-only ep2 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep2 | heldout_docs | judge_lenient | 0.477 | 0.439 | -0.038 [-0.063, -0.013] | 0.3408 |
| Qwen3-8B | QA-only ep2 | heldout_docs | keyfact_recall | 0.323 | 0.300 | -0.024 [-0.045, -0.002] | 1.0000 |
| Qwen3-8B | QA-only ep2 | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep2 | indomain | judge_lenient | 0.508 | 0.495 | -0.013 [-0.041, +0.014] | 1.0000 |
| Qwen3-8B | QA-only ep2 | indomain | keyfact_recall | 0.381 | 0.361 | -0.020 [-0.043, +0.005] | 1.0000 |
| Qwen3-8B | QA-only ep2 | seen_facts | exact_repro | 0.025 | 0.061 | +0.036 [+0.011, +0.061] | 0.5097 |
| Qwen3-8B | QA-only ep2 | seen_facts | judge_lenient | 0.590 | 0.627 | +0.038 [+0.005, +0.070] | 1.0000 |
| Qwen3-8B | QA-only ep2 | seen_facts | keyfact_recall | 0.426 | 0.404 | -0.022 [-0.056, +0.013] | 1.0000 |
| Qwen3-8B | QA-only ep2 | trained_exact | exact_repro | 0.184 | 0.496 | +0.312 [+0.268, +0.356] | 0.0365 |
| Qwen3-8B | QA-only ep2 | trained_exact | judge_lenient | 0.667 | 0.761 | +0.094 [+0.063, +0.125] | 0.0365 |
| Qwen3-8B | QA-only ep2 | trained_exact | keyfact_recall | 0.511 | 0.640 | +0.129 [+0.098, +0.163] | 0.0365 |
| Qwen3-8B | QA-only ep3 | heldout_docs | exact_repro | 0.002 | 0.000 | -0.002 [-0.006, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep3 | heldout_docs | judge_lenient | 0.476 | 0.431 | -0.044 [-0.070, -0.018] | 0.1299 |
| Qwen3-8B | QA-only ep3 | heldout_docs | keyfact_recall | 0.337 | 0.303 | -0.034 [-0.057, -0.012] | 0.1889 |
| Qwen3-8B | QA-only ep3 | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep3 | indomain | judge_lenient | 0.536 | 0.494 | -0.042 [-0.068, -0.014] | 0.1919 |
| Qwen3-8B | QA-only ep3 | indomain | keyfact_recall | 0.385 | 0.365 | -0.020 [-0.044, +0.004] | 1.0000 |
| Qwen3-8B | QA-only ep3 | seen_facts | exact_repro | 0.025 | 0.050 | +0.025 [+0.007, +0.047] | 1.0000 |
| Qwen3-8B | QA-only ep3 | seen_facts | judge_lenient | 0.584 | 0.619 | +0.032 [-0.007, +0.068] | 1.0000 |
| Qwen3-8B | QA-only ep3 | seen_facts | keyfact_recall | 0.407 | 0.402 | -0.004 [-0.040, +0.030] | 1.0000 |
| Qwen3-8B | QA-only ep3 | trained_exact | exact_repro | 0.214 | 0.610 | +0.396 [+0.350, +0.440] | 0.0365 |
| Qwen3-8B | QA-only ep3 | trained_exact | judge_lenient | 0.670 | 0.835 | +0.165 [+0.133, +0.195] | 0.0365 |
| Qwen3-8B | QA-only ep3 | trained_exact | keyfact_recall | 0.529 | 0.696 | +0.167 [+0.135, +0.201] | 0.0365 |
| Qwen3-8B | QA-only 1-epoch run | heldout_docs | exact_repro | 0.008 | 0.004 | -0.004 [-0.010, +0.000] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | heldout_docs | judge_lenient | 0.485 | 0.489 | +0.004 [-0.017, +0.027] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | heldout_docs | keyfact_recall | 0.341 | 0.335 | -0.005 [-0.026, +0.015] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | indomain | exact_repro | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | indomain | judge_lenient | 0.534 | 0.531 | -0.003 [-0.026, +0.021] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | indomain | keyfact_recall | 0.387 | 0.375 | -0.012 [-0.031, +0.007] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | seen_facts | exact_repro | 0.011 | 0.007 | -0.004 [-0.011, +0.000] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | seen_facts | judge_lenient | 0.561 | 0.543 | -0.018 [-0.050, +0.014] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | seen_facts | keyfact_recall | 0.352 | 0.346 | -0.007 [-0.032, +0.020] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | trained_exact | exact_repro | 0.016 | 0.022 | +0.006 [-0.002, +0.014] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | trained_exact | judge_lenient | 0.546 | 0.557 | +0.012 [-0.012, +0.036] | 1.0000 |
| Qwen3-8B | QA-only 1-epoch run | trained_exact | keyfact_recall | 0.361 | 0.377 | +0.016 [-0.004, +0.036] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | heldout_docs | exact_repro | 0.002 | 0.002 | +0.000 [+0.000, +0.000] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | heldout_docs | judge_lenient | 0.471 | 0.474 | +0.003 [-0.021, +0.027] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | heldout_docs | keyfact_recall | 0.320 | 0.306 | -0.014 [-0.035, +0.006] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | indomain | exact_repro | 0.002 | 0.004 | +0.002 [+0.000, +0.006] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | indomain | judge_lenient | 0.553 | 0.527 | -0.026 [-0.051, -0.001] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | indomain | keyfact_recall | 0.381 | 0.365 | -0.016 [-0.037, +0.005] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | seen_facts | exact_repro | 0.007 | 0.025 | +0.018 [+0.004, +0.036] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | seen_facts | judge_lenient | 0.604 | 0.597 | -0.007 [-0.041, +0.027] | 1.0000 |
| Qwen3-8B | QA-only ep1, seed 43 | seen_facts | keyfact_recall | 0.340 | 0.360 | +0.021 [-0.009, +0.052] | 1.0000 |

Same adapters, same pre-tokenized greedy decoding except the base weights: bf16 (earlier results) vs the NF4-dequantized bf16 weights the adapter was trained on. Paired bootstrap on identical items.
