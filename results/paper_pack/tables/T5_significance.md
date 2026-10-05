**Table 5. Fine-tuned vs same-family base on identical items**

| Metric | Split | Model | Variant | Δ | 95% CI | p | p (Holm) | Effect |
|---|---|---|---|---|---|---|---|---|
| judge acc | indomain | Qwen3-8B | 1-epoch (run 10-03) | -0.135 | [-0.165, -0.104] | 0.0005 | 0.0180 | -0.39 |
| judge acc | indomain | Qwen3-8B | ep1 | -0.134 | [-0.164, -0.104] | 0.0005 | 0.0180 | -0.39 |
| judge acc | indomain | Qwen3-8B | ep2 | -0.161 | [-0.192, -0.130] | 0.0005 | 0.0180 | -0.45 |
| judge acc | indomain | Qwen3-8B | ep3 | -0.133 | [-0.165, -0.103] | 0.0005 | 0.0180 | -0.38 |
| judge acc | indomain | Gemma 4 E4B | 1-epoch (run 10-03) | -0.187 | [-0.219, -0.157] | 0.0005 | 0.0180 | -0.53 |
| judge acc | indomain | Gemma 4 E4B | ep1 | -0.177 | [-0.210, -0.144] | 0.0005 | 0.0180 | -0.48 |
| judge acc | indomain | Gemma 4 E4B | ep2 | -0.200 | [-0.231, -0.168] | 0.0005 | 0.0180 | -0.56 |
| judge acc | indomain | Gemma 4 E4B | ep3 | -0.198 | [-0.230, -0.165] | 0.0005 | 0.0180 | -0.54 |
| judge acc | indomain | Llama 3.1 8B | 1-epoch (run 10-03) | -0.055 | [-0.088, -0.022] | 0.0010 | 0.0180 | -0.15 |
| judge acc | indomain | Llama 3.1 8B | ep1 | -0.050 | [-0.081, -0.018] | 0.0025 | 0.0180 | -0.14 |
| judge acc | indomain | Llama 3.1 8B | ep2 | -0.079 | [-0.111, -0.047] | 0.0005 | 0.0180 | -0.21 |
| judge acc | indomain | Llama 3.1 8B | ep3 | -0.061 | [-0.094, -0.028] | 0.0010 | 0.0180 | -0.16 |
| judge acc | heldout_docs | Qwen3-8B | 1-epoch (run 10-03) | -0.156 | [-0.186, -0.125] | 0.0005 | 0.0180 | -0.46 |
| judge acc | heldout_docs | Qwen3-8B | ep1 | -0.136 | [-0.166, -0.107] | 0.0005 | 0.0180 | -0.40 |
| judge acc | heldout_docs | Qwen3-8B | ep2 | -0.165 | [-0.196, -0.135] | 0.0005 | 0.0180 | -0.47 |
| judge acc | heldout_docs | Qwen3-8B | ep3 | -0.167 | [-0.197, -0.136] | 0.0005 | 0.0180 | -0.48 |
| judge acc | heldout_docs | Gemma 4 E4B | 1-epoch (run 10-03) | -0.161 | [-0.190, -0.132] | 0.0005 | 0.0180 | -0.47 |
| judge acc | heldout_docs | Gemma 4 E4B | ep1 | -0.179 | [-0.209, -0.150] | 0.0005 | 0.0180 | -0.52 |
| judge acc | heldout_docs | Gemma 4 E4B | ep2 | -0.181 | [-0.211, -0.153] | 0.0005 | 0.0180 | -0.54 |
| judge acc | heldout_docs | Gemma 4 E4B | ep3 | -0.195 | [-0.223, -0.165] | 0.0005 | 0.0180 | -0.58 |
| judge acc | heldout_docs | Llama 3.1 8B | 1-epoch (run 10-03) | -0.050 | [-0.080, -0.022] | 0.0015 | 0.0180 | -0.15 |
| judge acc | heldout_docs | Llama 3.1 8B | ep1 | -0.073 | [-0.107, -0.043] | 0.0005 | 0.0180 | -0.20 |
| judge acc | heldout_docs | Llama 3.1 8B | ep2 | -0.070 | [-0.102, -0.040] | 0.0005 | 0.0180 | -0.20 |
| judge acc | heldout_docs | Llama 3.1 8B | ep3 | -0.071 | [-0.101, -0.042] | 0.0005 | 0.0180 | -0.21 |
| judge acc | seen_facts | Qwen3-8B | 1-epoch (run 10-03) | -0.088 | [-0.129, -0.047] | 0.0005 | 0.0180 | -0.25 |
| judge acc | seen_facts | Qwen3-8B | ep1 | -0.075 | [-0.115, -0.034] | 0.0005 | 0.0180 | -0.21 |
| judge acc | seen_facts | Qwen3-8B | ep2 | -0.059 | [-0.100, -0.016] | 0.0055 | 0.0180 | -0.16 |
| judge acc | seen_facts | Qwen3-8B | ep3 | -0.065 | [-0.109, -0.018] | 0.0090 | 0.0180 | -0.17 |
| judge acc | seen_facts | Gemma 4 E4B | 1-epoch (run 10-03) | -0.084 | [-0.125, -0.039] | 0.0005 | 0.0180 | -0.22 |
| judge acc | seen_facts | Gemma 4 E4B | ep1 | -0.073 | [-0.118, -0.027] | 0.0010 | 0.0180 | -0.19 |
| judge acc | seen_facts | Gemma 4 E4B | ep2 | -0.070 | [-0.115, -0.023] | 0.0040 | 0.0180 | -0.18 |
| judge acc | seen_facts | Gemma 4 E4B | ep3 | -0.039 | [-0.082, +0.005] | 0.0810 | 0.0810 | -0.10 |
| judge acc | seen_facts | Llama 3.1 8B | 1-epoch (run 10-03) | +0.072 | [+0.027, +0.115] | 0.0025 | 0.0180 | +0.20 |
| judge acc | seen_facts | Llama 3.1 8B | ep1 | +0.093 | [+0.048, +0.138] | 0.0005 | 0.0180 | +0.24 |
| judge acc | seen_facts | Llama 3.1 8B | ep2 | +0.152 | [+0.102, +0.201] | 0.0005 | 0.0180 | +0.36 |
| judge acc | seen_facts | Llama 3.1 8B | ep3 | +0.133 | [+0.086, +0.179] | 0.0005 | 0.0180 | +0.33 |
| KF recall | indomain | Qwen3-8B | 1-epoch (run 10-03) | -0.177 | [-0.205, -0.149] | 0.0005 | 0.0180 | -0.56 |
| KF recall | indomain | Qwen3-8B | ep1 | -0.204 | [-0.231, -0.176] | 0.0005 | 0.0180 | -0.66 |
| KF recall | indomain | Qwen3-8B | ep2 | -0.184 | [-0.214, -0.152] | 0.0005 | 0.0180 | -0.54 |
| KF recall | indomain | Qwen3-8B | ep3 | -0.179 | [-0.208, -0.149] | 0.0005 | 0.0180 | -0.54 |
| KF recall | indomain | Gemma 4 E4B | 1-epoch (run 10-03) | -0.195 | [-0.224, -0.167] | 0.0005 | 0.0180 | -0.57 |
| KF recall | indomain | Gemma 4 E4B | ep1 | -0.203 | [-0.234, -0.173] | 0.0005 | 0.0180 | -0.58 |
| KF recall | indomain | Gemma 4 E4B | ep2 | -0.210 | [-0.240, -0.180] | 0.0005 | 0.0180 | -0.60 |
| KF recall | indomain | Gemma 4 E4B | ep3 | -0.206 | [-0.234, -0.177] | 0.0005 | 0.0180 | -0.61 |
| KF recall | indomain | Llama 3.1 8B | 1-epoch (run 10-03) | -0.180 | [-0.209, -0.151] | 0.0005 | 0.0180 | -0.54 |
| KF recall | indomain | Llama 3.1 8B | ep1 | -0.183 | [-0.213, -0.152] | 0.0005 | 0.0180 | -0.52 |
| KF recall | indomain | Llama 3.1 8B | ep2 | -0.184 | [-0.211, -0.155] | 0.0005 | 0.0180 | -0.55 |
| KF recall | indomain | Llama 3.1 8B | ep3 | -0.184 | [-0.213, -0.152] | 0.0005 | 0.0180 | -0.53 |
| KF recall | heldout_docs | Qwen3-8B | 1-epoch (run 10-03) | -0.177 | [-0.205, -0.150] | 0.0005 | 0.0180 | -0.58 |
| KF recall | heldout_docs | Qwen3-8B | ep1 | -0.172 | [-0.202, -0.145] | 0.0005 | 0.0180 | -0.52 |
| KF recall | heldout_docs | Qwen3-8B | ep2 | -0.194 | [-0.223, -0.165] | 0.0005 | 0.0180 | -0.59 |
| KF recall | heldout_docs | Qwen3-8B | ep3 | -0.181 | [-0.209, -0.154] | 0.0005 | 0.0180 | -0.57 |
| KF recall | heldout_docs | Gemma 4 E4B | 1-epoch (run 10-03) | -0.183 | [-0.211, -0.154] | 0.0005 | 0.0180 | -0.55 |
| KF recall | heldout_docs | Gemma 4 E4B | ep1 | -0.219 | [-0.249, -0.189] | 0.0005 | 0.0180 | -0.64 |
| KF recall | heldout_docs | Gemma 4 E4B | ep2 | -0.213 | [-0.243, -0.184] | 0.0005 | 0.0180 | -0.64 |
| KF recall | heldout_docs | Gemma 4 E4B | ep3 | -0.205 | [-0.234, -0.176] | 0.0005 | 0.0180 | -0.62 |
| KF recall | heldout_docs | Llama 3.1 8B | 1-epoch (run 10-03) | -0.161 | [-0.189, -0.134] | 0.0005 | 0.0180 | -0.52 |
| KF recall | heldout_docs | Llama 3.1 8B | ep1 | -0.169 | [-0.198, -0.140] | 0.0005 | 0.0180 | -0.52 |
| KF recall | heldout_docs | Llama 3.1 8B | ep2 | -0.182 | [-0.210, -0.155] | 0.0005 | 0.0180 | -0.58 |
| KF recall | heldout_docs | Llama 3.1 8B | ep3 | -0.185 | [-0.213, -0.155] | 0.0005 | 0.0180 | -0.57 |
| KF recall | seen_facts | Qwen3-8B | 1-epoch (run 10-03) | -0.170 | [-0.211, -0.131] | 0.0005 | 0.0180 | -0.49 |
| KF recall | seen_facts | Qwen3-8B | ep1 | -0.185 | [-0.223, -0.144] | 0.0005 | 0.0180 | -0.54 |
| KF recall | seen_facts | Qwen3-8B | ep2 | -0.097 | [-0.137, -0.057] | 0.0005 | 0.0180 | -0.28 |
| KF recall | seen_facts | Qwen3-8B | ep3 | -0.116 | [-0.156, -0.075] | 0.0005 | 0.0180 | -0.32 |
| KF recall | seen_facts | Gemma 4 E4B | 1-epoch (run 10-03) | -0.180 | [-0.222, -0.138] | 0.0005 | 0.0180 | -0.50 |
| KF recall | seen_facts | Gemma 4 E4B | ep1 | -0.171 | [-0.217, -0.125] | 0.0005 | 0.0180 | -0.43 |
| KF recall | seen_facts | Gemma 4 E4B | ep2 | -0.144 | [-0.188, -0.099] | 0.0005 | 0.0180 | -0.37 |
| KF recall | seen_facts | Gemma 4 E4B | ep3 | -0.128 | [-0.172, -0.083] | 0.0005 | 0.0180 | -0.33 |
| KF recall | seen_facts | Llama 3.1 8B | 1-epoch (run 10-03) | -0.140 | [-0.180, -0.098] | 0.0005 | 0.0180 | -0.40 |
| KF recall | seen_facts | Llama 3.1 8B | ep1 | -0.110 | [-0.155, -0.067] | 0.0005 | 0.0180 | -0.30 |
| KF recall | seen_facts | Llama 3.1 8B | ep2 | -0.010 | [-0.061, +0.038] | 0.6802 | 1.0000 | -0.02 |
| KF recall | seen_facts | Llama 3.1 8B | ep3 | -0.013 | [-0.059, +0.032] | 0.5977 | 1.0000 | -0.03 |
| contra. | indomain | Qwen3-8B | 1-epoch (run 10-03) | -0.023 | [-0.044, -0.003] | 0.0235 | 0.7751 | -0.10 |
| contra. | indomain | Qwen3-8B | ep1 | -0.026 | [-0.047, -0.005] | 0.0170 | 0.5777 | -0.11 |
| contra. | indomain | Qwen3-8B | ep2 | -0.011 | [-0.031, +0.009] | 0.2884 | 1.0000 | -0.05 |
| contra. | indomain | Qwen3-8B | ep3 | -0.017 | [-0.039, +0.003] | 0.1264 | 1.0000 | -0.07 |
| contra. | indomain | Gemma 4 E4B | 1-epoch (run 10-03) | -0.007 | [-0.027, +0.013] | 0.5167 | 1.0000 | -0.03 |
| contra. | indomain | Gemma 4 E4B | ep1 | -0.009 | [-0.030, +0.011] | 0.3918 | 1.0000 | -0.04 |
| contra. | indomain | Gemma 4 E4B | ep2 | -0.008 | [-0.029, +0.011] | 0.4268 | 1.0000 | -0.04 |
| contra. | indomain | Gemma 4 E4B | ep3 | -0.001 | [-0.023, +0.020] | 0.9010 | 1.0000 | -0.01 |
| contra. | indomain | Llama 3.1 8B | 1-epoch (run 10-03) | -0.002 | [-0.020, +0.017] | 0.8466 | 1.0000 | -0.01 |
| contra. | indomain | Llama 3.1 8B | ep1 | -0.003 | [-0.023, +0.015] | 0.7291 | 1.0000 | -0.02 |
| contra. | indomain | Llama 3.1 8B | ep2 | +0.002 | [-0.017, +0.020] | 0.8166 | 1.0000 | +0.01 |
| contra. | indomain | Llama 3.1 8B | ep3 | -0.014 | [-0.032, +0.005] | 0.1459 | 1.0000 | -0.06 |
| contra. | heldout_docs | Qwen3-8B | 1-epoch (run 10-03) | +0.017 | [+0.001, +0.034] | 0.0380 | 1.0000 | +0.09 |
| contra. | heldout_docs | Qwen3-8B | ep1 | +0.013 | [-0.004, +0.031] | 0.1364 | 1.0000 | +0.06 |
| contra. | heldout_docs | Qwen3-8B | ep2 | -0.003 | [-0.023, +0.016] | 0.7191 | 1.0000 | -0.02 |
| contra. | heldout_docs | Qwen3-8B | ep3 | +0.003 | [-0.017, +0.021] | 0.7791 | 1.0000 | +0.01 |
| contra. | heldout_docs | Gemma 4 E4B | 1-epoch (run 10-03) | +0.017 | [-0.002, +0.037] | 0.0790 | 1.0000 | +0.08 |
| contra. | heldout_docs | Gemma 4 E4B | ep1 | +0.031 | [+0.012, +0.050] | 0.0020 | 0.0720 | +0.14 |
| contra. | heldout_docs | Gemma 4 E4B | ep2 | +0.028 | [+0.008, +0.049] | 0.0080 | 0.2799 | +0.12 |
| contra. | heldout_docs | Gemma 4 E4B | ep3 | +0.019 | [+0.000, +0.040] | 0.0550 | 1.0000 | +0.09 |
| contra. | heldout_docs | Llama 3.1 8B | 1-epoch (run 10-03) | +0.002 | [-0.019, +0.022] | 0.8576 | 1.0000 | +0.01 |
| contra. | heldout_docs | Llama 3.1 8B | ep1 | -0.007 | [-0.027, +0.013] | 0.4658 | 1.0000 | -0.03 |
| contra. | heldout_docs | Llama 3.1 8B | ep2 | +0.002 | [-0.017, +0.021] | 0.8366 | 1.0000 | +0.01 |
| contra. | heldout_docs | Llama 3.1 8B | ep3 | -0.010 | [-0.030, +0.010] | 0.3273 | 1.0000 | -0.04 |
| contra. | seen_facts | Qwen3-8B | 1-epoch (run 10-03) | -0.016 | [-0.045, +0.014] | 0.2849 | 1.0000 | -0.06 |
| contra. | seen_facts | Qwen3-8B | ep1 | -0.022 | [-0.053, +0.008] | 0.1589 | 1.0000 | -0.09 |
| contra. | seen_facts | Qwen3-8B | ep2 | -0.025 | [-0.053, +0.001] | 0.0670 | 1.0000 | -0.11 |
| contra. | seen_facts | Qwen3-8B | ep3 | -0.030 | [-0.057, -0.003] | 0.0305 | 0.9755 | -0.13 |
| contra. | seen_facts | Gemma 4 E4B | 1-epoch (run 10-03) | +0.018 | [-0.010, +0.046] | 0.2059 | 1.0000 | +0.08 |
| contra. | seen_facts | Gemma 4 E4B | ep1 | +0.022 | [-0.007, +0.051] | 0.1349 | 1.0000 | +0.09 |
| contra. | seen_facts | Gemma 4 E4B | ep2 | +0.027 | [-0.002, +0.058] | 0.0795 | 1.0000 | +0.11 |
| contra. | seen_facts | Gemma 4 E4B | ep3 | +0.023 | [-0.006, +0.051] | 0.1139 | 1.0000 | +0.09 |
| contra. | seen_facts | Llama 3.1 8B | 1-epoch (run 10-03) | +0.016 | [-0.013, +0.045] | 0.2749 | 1.0000 | +0.07 |
| contra. | seen_facts | Llama 3.1 8B | ep1 | -0.005 | [-0.033, +0.023] | 0.7486 | 1.0000 | -0.02 |
| contra. | seen_facts | Llama 3.1 8B | ep2 | -0.019 | [-0.044, +0.007] | 0.1609 | 1.0000 | -0.09 |
| contra. | seen_facts | Llama 3.1 8B | ep3 | -0.025 | [-0.053, +0.001] | 0.0705 | 1.0000 | -0.11 |
| token F1 | indomain | Qwen3-8B | 1-epoch (run 10-03) | +0.111 | [+0.100, +0.122] | 0.0005 | 0.0180 | +0.89 |
| token F1 | indomain | Qwen3-8B | ep1 | +0.098 | [+0.087, +0.109] | 0.0005 | 0.0180 | +0.77 |
| token F1 | indomain | Qwen3-8B | ep2 | +0.078 | [+0.068, +0.088] | 0.0005 | 0.0180 | +0.67 |
| token F1 | indomain | Qwen3-8B | ep3 | +0.078 | [+0.068, +0.088] | 0.0005 | 0.0180 | +0.65 |
| token F1 | indomain | Gemma 4 E4B | 1-epoch (run 10-03) | +0.201 | [+0.190, +0.213] | 0.0005 | 0.0180 | +1.51 |
| token F1 | indomain | Gemma 4 E4B | ep1 | +0.192 | [+0.180, +0.204] | 0.0005 | 0.0180 | +1.38 |
| token F1 | indomain | Gemma 4 E4B | ep2 | +0.174 | [+0.163, +0.185] | 0.0005 | 0.0180 | +1.42 |
| token F1 | indomain | Gemma 4 E4B | ep3 | +0.163 | [+0.152, +0.174] | 0.0005 | 0.0180 | +1.29 |
| token F1 | indomain | Llama 3.1 8B | 1-epoch (run 10-03) | +0.190 | [+0.180, +0.201] | 0.0005 | 0.0180 | +1.54 |
| token F1 | indomain | Llama 3.1 8B | ep1 | +0.176 | [+0.165, +0.187] | 0.0005 | 0.0180 | +1.41 |
| token F1 | indomain | Llama 3.1 8B | ep2 | +0.155 | [+0.145, +0.166] | 0.0005 | 0.0180 | +1.31 |
| token F1 | indomain | Llama 3.1 8B | ep3 | +0.152 | [+0.142, +0.162] | 0.0005 | 0.0180 | +1.32 |
| token F1 | heldout_docs | Qwen3-8B | 1-epoch (run 10-03) | +0.095 | [+0.085, +0.106] | 0.0005 | 0.0180 | +0.81 |
| token F1 | heldout_docs | Qwen3-8B | ep1 | +0.081 | [+0.071, +0.091] | 0.0005 | 0.0180 | +0.72 |
| token F1 | heldout_docs | Qwen3-8B | ep2 | +0.060 | [+0.050, +0.070] | 0.0005 | 0.0180 | +0.54 |
| token F1 | heldout_docs | Qwen3-8B | ep3 | +0.056 | [+0.046, +0.065] | 0.0005 | 0.0180 | +0.51 |
| token F1 | heldout_docs | Gemma 4 E4B | 1-epoch (run 10-03) | +0.183 | [+0.173, +0.194] | 0.0005 | 0.0180 | +1.49 |
| token F1 | heldout_docs | Gemma 4 E4B | ep1 | +0.170 | [+0.159, +0.182] | 0.0005 | 0.0180 | +1.33 |
| token F1 | heldout_docs | Gemma 4 E4B | ep2 | +0.155 | [+0.144, +0.165] | 0.0005 | 0.0180 | +1.26 |
| token F1 | heldout_docs | Gemma 4 E4B | ep3 | +0.149 | [+0.138, +0.159] | 0.0005 | 0.0180 | +1.22 |
| token F1 | heldout_docs | Llama 3.1 8B | 1-epoch (run 10-03) | +0.166 | [+0.156, +0.177] | 0.0005 | 0.0180 | +1.34 |
| token F1 | heldout_docs | Llama 3.1 8B | ep1 | +0.155 | [+0.144, +0.166] | 0.0005 | 0.0180 | +1.28 |
| token F1 | heldout_docs | Llama 3.1 8B | ep2 | +0.131 | [+0.122, +0.141] | 0.0005 | 0.0180 | +1.19 |
| token F1 | heldout_docs | Llama 3.1 8B | ep3 | +0.132 | [+0.122, +0.142] | 0.0005 | 0.0180 | +1.19 |
| token F1 | seen_facts | Qwen3-8B | 1-epoch (run 10-03) | +0.113 | [+0.099, +0.128] | 0.0005 | 0.0180 | +0.92 |
| token F1 | seen_facts | Qwen3-8B | ep1 | +0.123 | [+0.108, +0.138] | 0.0005 | 0.0180 | +0.95 |
| token F1 | seen_facts | Qwen3-8B | ep2 | +0.153 | [+0.135, +0.171] | 0.0005 | 0.0180 | +1.01 |
| token F1 | seen_facts | Qwen3-8B | ep3 | +0.144 | [+0.126, +0.162] | 0.0005 | 0.0180 | +0.95 |
| token F1 | seen_facts | Gemma 4 E4B | 1-epoch (run 10-03) | +0.166 | [+0.153, +0.180] | 0.0005 | 0.0180 | +1.39 |
| token F1 | seen_facts | Gemma 4 E4B | ep1 | +0.189 | [+0.175, +0.205] | 0.0005 | 0.0180 | +1.48 |
| token F1 | seen_facts | Gemma 4 E4B | ep2 | +0.208 | [+0.194, +0.225] | 0.0005 | 0.0180 | +1.59 |
| token F1 | seen_facts | Gemma 4 E4B | ep3 | +0.218 | [+0.203, +0.234] | 0.0005 | 0.0180 | +1.60 |
| token F1 | seen_facts | Llama 3.1 8B | 1-epoch (run 10-03) | +0.187 | [+0.172, +0.202] | 0.0005 | 0.0180 | +1.46 |
| token F1 | seen_facts | Llama 3.1 8B | ep1 | +0.209 | [+0.191, +0.228] | 0.0005 | 0.0180 | +1.33 |
| token F1 | seen_facts | Llama 3.1 8B | ep2 | +0.309 | [+0.285, +0.335] | 0.0005 | 0.0180 | +1.37 |
| token F1 | seen_facts | Llama 3.1 8B | ep3 | +0.319 | [+0.292, +0.344] | 0.0005 | 0.0180 | +1.38 |
| strict acc (McNemar) | indomain | Qwen3-8B | 1-epoch (run 10-03) | -0.230 | – | 0.0000 | 0.0000 | -0.34 |
| strict acc (McNemar) | indomain | Qwen3-8B | ep1 | -0.230 | – | 0.0000 | 0.0000 | -0.35 |
| strict acc (McNemar) | indomain | Qwen3-8B | ep2 | -0.268 | – | 0.0000 | 0.0000 | -0.38 |
| strict acc (McNemar) | indomain | Qwen3-8B | ep3 | -0.244 | – | 0.0000 | 0.0000 | -0.34 |
| strict acc (McNemar) | indomain | Gemma 4 E4B | 1-epoch (run 10-03) | -0.327 | – | 0.0000 | 0.0000 | -0.40 |
| strict acc (McNemar) | indomain | Gemma 4 E4B | ep1 | -0.323 | – | 0.0000 | 0.0000 | -0.37 |
| strict acc (McNemar) | indomain | Gemma 4 E4B | ep2 | -0.339 | – | 0.0000 | 0.0000 | -0.39 |
| strict acc (McNemar) | indomain | Gemma 4 E4B | ep3 | -0.339 | – | 0.0000 | 0.0000 | -0.39 |
| strict acc (McNemar) | indomain | Llama 3.1 8B | 1-epoch (run 10-03) | -0.120 | – | 0.0000 | 0.0000 | -0.21 |
| strict acc (McNemar) | indomain | Llama 3.1 8B | ep1 | -0.132 | – | 0.0000 | 0.0000 | -0.23 |
| strict acc (McNemar) | indomain | Llama 3.1 8B | ep2 | -0.166 | – | 0.0000 | 0.0000 | -0.27 |
| strict acc (McNemar) | indomain | Llama 3.1 8B | ep3 | -0.150 | – | 0.0000 | 0.0000 | -0.23 |
| strict acc (McNemar) | heldout_docs | Qwen3-8B | 1-epoch (run 10-03) | -0.242 | – | 0.0000 | 0.0000 | -0.37 |
| strict acc (McNemar) | heldout_docs | Qwen3-8B | ep1 | -0.234 | – | 0.0000 | 0.0000 | -0.37 |
| strict acc (McNemar) | heldout_docs | Qwen3-8B | ep2 | -0.256 | – | 0.0000 | 0.0000 | -0.39 |
| strict acc (McNemar) | heldout_docs | Qwen3-8B | ep3 | -0.261 | – | 0.0000 | 0.0000 | -0.41 |
| strict acc (McNemar) | heldout_docs | Gemma 4 E4B | 1-epoch (run 10-03) | -0.284 | – | 0.0000 | 0.0000 | -0.39 |
| strict acc (McNemar) | heldout_docs | Gemma 4 E4B | ep1 | -0.302 | – | 0.0000 | 0.0000 | -0.40 |
| strict acc (McNemar) | heldout_docs | Gemma 4 E4B | ep2 | -0.292 | – | 0.0000 | 0.0000 | -0.42 |
| strict acc (McNemar) | heldout_docs | Gemma 4 E4B | ep3 | -0.298 | – | 0.0000 | 0.0000 | -0.44 |
| strict acc (McNemar) | heldout_docs | Llama 3.1 8B | 1-epoch (run 10-03) | -0.129 | – | 0.0000 | 0.0000 | -0.25 |
| strict acc (McNemar) | heldout_docs | Llama 3.1 8B | ep1 | -0.141 | – | 0.0000 | 0.0000 | -0.26 |
| strict acc (McNemar) | heldout_docs | Llama 3.1 8B | ep2 | -0.151 | – | 0.0000 | 0.0000 | -0.28 |
| strict acc (McNemar) | heldout_docs | Llama 3.1 8B | ep3 | -0.149 | – | 0.0000 | 0.0000 | -0.28 |
| strict acc (McNemar) | seen_facts | Qwen3-8B | 1-epoch (run 10-03) | -0.154 | – | 0.0000 | 0.0000 | -0.27 |
| strict acc (McNemar) | seen_facts | Qwen3-8B | ep1 | -0.158 | – | 0.0000 | 0.0000 | -0.26 |
| strict acc (McNemar) | seen_facts | Qwen3-8B | ep2 | -0.161 | – | 0.0000 | 0.0000 | -0.25 |
| strict acc (McNemar) | seen_facts | Qwen3-8B | ep3 | -0.154 | – | 0.0000 | 0.0001 | -0.20 |
| strict acc (McNemar) | seen_facts | Gemma 4 E4B | 1-epoch (run 10-03) | -0.197 | – | 0.0000 | 0.0000 | -0.28 |
| strict acc (McNemar) | seen_facts | Gemma 4 E4B | ep1 | -0.172 | – | 0.0000 | 0.0000 | -0.24 |
| strict acc (McNemar) | seen_facts | Gemma 4 E4B | ep2 | -0.165 | – | 0.0000 | 0.0000 | -0.23 |
| strict acc (McNemar) | seen_facts | Gemma 4 E4B | ep3 | -0.140 | – | 0.0001 | 0.0002 | -0.21 |
| strict acc (McNemar) | seen_facts | Llama 3.1 8B | 1-epoch (run 10-03) | +0.079 | – | 0.0246 | 0.0246 | +0.12 |
| strict acc (McNemar) | seen_facts | Llama 3.1 8B | ep1 | +0.111 | – | 0.0017 | 0.0034 | +0.17 |
| strict acc (McNemar) | seen_facts | Llama 3.1 8B | ep2 | +0.197 | – | 0.0000 | 0.0000 | +0.24 |
| strict acc (McNemar) | seen_facts | Llama 3.1 8B | ep3 | +0.179 | – | 0.0000 | 0.0000 | +0.23 |

Paired bootstrap (2,000 resamples; effect = Cohen's d_z) and exact McNemar on binary correct (effect = Cohen's g). Holm-Bonferroni within each metric across 36 comparisons (12 systems × 3 splits). Full list: results/local_eval/significance.csv.
