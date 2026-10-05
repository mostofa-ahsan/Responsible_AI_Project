**Table 7b. Paraphrased training questions (seen facts, n = 279), all variants**

| Model | Variant | Judge acc | KF recall | Contra. | F1 | Exact repro. | ROUGE-L | Words |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.649 | 0.523 | 0.108 | 0.213 | 0.000 | 0.141 | 141.8 |
| Qwen3-8B | concise base (<= 40 words) | 0.573† | 0.301† | 0.107 | 0.252† | 0.000 | 0.183† | 24.5† |
| Qwen3-8B | 1-epoch (run 10-03) | 0.561† | 0.352† | 0.092 | 0.326† | 0.011 | 0.240† | 37.2† |
| Qwen3-8B | ep1 | 0.573† | 0.338† | 0.086 | 0.336† | 0.011 | 0.255† | 34.6† |
| Qwen3-8B | ep2 | 0.590† | 0.426† | 0.083 | 0.366† | 0.025 | 0.283† | 42.3† |
| Qwen3-8B | ep3 | 0.584 | 0.407† | 0.078 | 0.357† | 0.025 | 0.279† | 43.0† |
| Qwen3-8B | NF4 base + adapter (ep1) | 0.597 | 0.338† | 0.070 | 0.351† | 0.014 | 0.270† | 33.5† |
| Qwen3-8B | merged 4-bit (AWQ/W4A16, vLLM) (ep1) | 0.556† | 0.324† | 0.106 | 0.328† | 0.000 | 0.248† | 37.0† |
| Gemma 4 E4B | base | 0.668 | 0.530 | 0.063 | 0.151 | 0.000 | 0.101 | 237.5 |
| Gemma 4 E4B | concise base (<= 40 words) | 0.600† | 0.311† | 0.086 | 0.250† | 0.000 | 0.181† | 28.1† |
| Gemma 4 E4B | 1-epoch (run 10-03) | 0.584† | 0.350† | 0.080 | 0.316† | 0.004 | 0.236† | 34.5† |
| Gemma 4 E4B | ep1 | 0.595† | 0.359† | 0.085 | 0.340† | 0.014 | 0.256† | 34.3† |
| Gemma 4 E4B | ep2 | 0.599† | 0.386† | 0.090 | 0.359† | 0.018 | 0.272† | 36.6† |
| Gemma 4 E4B | ep3 | 0.629 | 0.402† | 0.086 | 0.368† | 0.018 | 0.279† | 37.6† |
| Gemma 4 E4B | NF4 base + adapter (ep1) | 0.573† | 0.353† | 0.087 | 0.341† | 0.014 | 0.263† | 34.2† |
| Gemma 4 E4B | merged 4-bit (AWQ/W4A16, vLLM) (ep1) | 0.550† | 0.328† | 0.088 | 0.322† | 0.004 | 0.238† | 36.1† |
| Llama 3.1 8B | base | 0.522 | 0.498 | 0.083 | 0.154 | 0.000 | 0.107 | 227.9 |
| Llama 3.1 8B | concise base (<= 40 words) | 0.527 | 0.330† | 0.119 | 0.258† | 0.000 | 0.182† | 39.3† |
| Llama 3.1 8B | 1-epoch (run 10-03) | 0.595† | 0.358† | 0.099 | 0.342† | 0.014 | 0.260† | 36.6† |
| Llama 3.1 8B | ep1 | 0.615† | 0.387† | 0.078 | 0.363† | 0.025 | 0.281† | 38.0† |
| Llama 3.1 8B | ep2 | 0.674† | 0.487 | 0.065 | 0.463† | 0.125† | 0.397† | 43.2† |
| Llama 3.1 8B | ep3 | 0.654† | 0.485 | 0.058 | 0.473† | 0.143† | 0.411† | 42.2† |
| Llama 3.1 8B | NF4 base + adapter (ep2) | **0.722†** | **0.539** | **0.043** | **0.542†** | **0.258†** | **0.487†** | 42.6† |
| Llama 3.1 8B | merged 4-bit (AWQ/W4A16, vLLM) (ep2) | 0.596† | 0.422† | 0.071 | 0.409† | 0.061† | 0.337† | 43.8† |

Judge acc = (correct + 0.5 partial) / N (local judge, prompt v1). KF recall = share of reference key facts supported by the answer (MiniCheck-7B). Exact repro. = share of answers with ROUGE-L ≥ 0.8 against the trained answer. † = differs from the same model's base (paired bootstrap, Holm-adjusted p < 0.05). Best per column in bold.
