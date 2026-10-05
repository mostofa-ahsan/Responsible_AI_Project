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
| KF recall | indomain | Qwen3-8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | indomain | Qwen3-8B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | indomain | Qwen3-8B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | indomain | Qwen3-8B | ep3 | +nan | – | nan | nan | +nan |
| KF recall | indomain | Gemma 4 E4B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | indomain | Gemma 4 E4B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | indomain | Gemma 4 E4B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | indomain | Gemma 4 E4B | ep3 | +nan | – | nan | nan | +nan |
| KF recall | indomain | Llama 3.1 8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | indomain | Llama 3.1 8B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | indomain | Llama 3.1 8B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | indomain | Llama 3.1 8B | ep3 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Qwen3-8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Qwen3-8B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Qwen3-8B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Qwen3-8B | ep3 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Gemma 4 E4B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Gemma 4 E4B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Gemma 4 E4B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Gemma 4 E4B | ep3 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Llama 3.1 8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Llama 3.1 8B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Llama 3.1 8B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | heldout_docs | Llama 3.1 8B | ep3 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Qwen3-8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Qwen3-8B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Qwen3-8B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Qwen3-8B | ep3 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Gemma 4 E4B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Gemma 4 E4B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Gemma 4 E4B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Gemma 4 E4B | ep3 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Llama 3.1 8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Llama 3.1 8B | ep1 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Llama 3.1 8B | ep2 | +nan | – | nan | nan | +nan |
| KF recall | seen_facts | Llama 3.1 8B | ep3 | +nan | – | nan | nan | +nan |
| contra. | indomain | Qwen3-8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | indomain | Qwen3-8B | ep1 | +nan | – | nan | nan | +nan |
| contra. | indomain | Qwen3-8B | ep2 | +nan | – | nan | nan | +nan |
| contra. | indomain | Qwen3-8B | ep3 | +nan | – | nan | nan | +nan |
| contra. | indomain | Gemma 4 E4B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | indomain | Gemma 4 E4B | ep1 | +nan | – | nan | nan | +nan |
| contra. | indomain | Gemma 4 E4B | ep2 | +nan | – | nan | nan | +nan |
| contra. | indomain | Gemma 4 E4B | ep3 | +nan | – | nan | nan | +nan |
| contra. | indomain | Llama 3.1 8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | indomain | Llama 3.1 8B | ep1 | +nan | – | nan | nan | +nan |
| contra. | indomain | Llama 3.1 8B | ep2 | +nan | – | nan | nan | +nan |
| contra. | indomain | Llama 3.1 8B | ep3 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Qwen3-8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Qwen3-8B | ep1 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Qwen3-8B | ep2 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Qwen3-8B | ep3 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Gemma 4 E4B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Gemma 4 E4B | ep1 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Gemma 4 E4B | ep2 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Gemma 4 E4B | ep3 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Llama 3.1 8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Llama 3.1 8B | ep1 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Llama 3.1 8B | ep2 | +nan | – | nan | nan | +nan |
| contra. | heldout_docs | Llama 3.1 8B | ep3 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Qwen3-8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Qwen3-8B | ep1 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Qwen3-8B | ep2 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Qwen3-8B | ep3 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Gemma 4 E4B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Gemma 4 E4B | ep1 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Gemma 4 E4B | ep2 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Gemma 4 E4B | ep3 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Llama 3.1 8B | 1-epoch (run 10-03) | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Llama 3.1 8B | ep1 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Llama 3.1 8B | ep2 | +nan | – | nan | nan | +nan |
| contra. | seen_facts | Llama 3.1 8B | ep3 | +nan | – | nan | nan | +nan |
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
