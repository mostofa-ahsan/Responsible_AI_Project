**Table 7c. Held-out documents (n = 500), all variants**

| Model | Variant | Judge acc | KF recall | Contra. | F1 | Exact repro. | ROUGE-L | Words |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.642 | 0.518 | 0.088 | 0.259 | 0.000 | 0.175 | 121.7 |
| Qwen3-8B | concise base (<= 40 words) | – | – | – | 0.283† | 0.004 | 0.213† | 24.8† |
| Qwen3-8B | 1-epoch (run 10-03) | 0.485† | 0.341† | 0.106 | 0.355† | **0.008** | 0.267† | 34.9† |
| Qwen3-8B | ep1 | 0.506† | 0.345† | 0.101 | 0.340† | 0.004 | 0.256† | 33.2† |
| Qwen3-8B | ep2 | 0.477† | 0.323† | 0.085 | 0.319† | 0.002 | 0.233† | 40.7† |
| Qwen3-8B | ep3 | 0.476† | 0.337† | 0.091 | 0.315† | 0.002 | 0.227† | 39.9† |
| Gemma 4 E4B | base | **0.679** | **0.541** | **0.065** | 0.177 | 0.000 | 0.118 | 222.4 |
| Gemma 4 E4B | 1-epoch (run 10-03) | 0.518† | 0.357† | 0.083 | **0.361†** | 0.004 | **0.276†** | 31.7† |
| Gemma 4 E4B | ep1 | 0.500† | 0.322† | 0.096† | 0.347† | 0.002 | 0.267† | 31.6† |
| Gemma 4 E4B | ep2 | 0.498† | 0.327† | 0.093 | 0.332† | 0.002 | 0.246† | 35.0† |
| Gemma 4 E4B | ep3 | 0.484† | 0.335† | 0.085 | 0.326† | 0.002 | 0.239† | 35.9† |
| Llama 3.1 8B | base | 0.534 | 0.493 | 0.100 | 0.179 | 0.000 | 0.127 | 212.8 |
| Llama 3.1 8B | 1-epoch (run 10-03) | 0.485† | 0.333† | 0.102 | 0.346† | 0.002 | 0.259† | 35.0† |
| Llama 3.1 8B | ep1 | 0.460† | 0.325† | 0.093 | 0.334† | 0.002 | 0.248† | 36.1† |
| Llama 3.1 8B | ep2 | 0.465† | 0.311† | 0.102 | 0.310† | 0.002 | 0.222† | 42.8† |
| Llama 3.1 8B | ep3 | 0.462† | 0.308† | 0.090 | 0.311† | 0.002 | 0.226† | 41.1† |

Judge acc = (correct + 0.5 partial) / N (local judge, prompt v1). KF recall = share of reference key facts supported by the answer (MiniCheck-7B). Exact repro. = share of answers with ROUGE-L ≥ 0.8 against the trained answer. † = differs from the same model's base (paired bootstrap, Holm-adjusted p < 0.05). Best per column in bold.
