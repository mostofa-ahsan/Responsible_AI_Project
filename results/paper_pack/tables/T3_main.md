**Table 3. Main results, In-domain (unseen chunks) (n = 500 per system)**

| Model | Variant | Judge acc | Halluc. | KF recall | Contra. | F1 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.669 | 0.360 | 0.564 | 0.096 | 0.265 | 0.181 |
| Qwen3-8B | 1-epoch (run 10-03) | 0.534† | 0.336 | 0.387† | 0.073 | 0.376† | 0.283† |
| Qwen3-8B | ep1 | 0.535† | 0.320 | 0.360† | 0.070 | 0.363† | 0.275† |
| Qwen3-8B | ep2 | 0.508† | 0.408 | 0.381† | 0.085 | 0.343† | 0.254† |
| Qwen3-8B | ep3 | 0.536† | 0.369 | 0.385† | 0.079 | 0.343† | 0.253† |
| Gemma 4 E4B | base | **0.731** | **0.176** | **0.574** | 0.078 | 0.177 | 0.120 |
| Gemma 4 E4B | 1-epoch (run 10-03) | 0.543† | 0.274† | 0.380† | 0.071 | **0.378†** | **0.291†** |
| Gemma 4 E4B | ep1 | 0.553† | 0.276† | 0.371† | **0.069** | 0.369† | 0.280† |
| Gemma 4 E4B | ep2 | 0.530† | 0.308† | 0.364† | 0.070 | 0.351† | 0.263† |
| Gemma 4 E4B | ep3 | 0.532† | 0.346† | 0.369† | 0.076 | 0.340† | 0.252† |
| Llama 3.1 8B | base | 0.575 | 0.433 | 0.555 | 0.091 | 0.182 | 0.130 |
| Llama 3.1 8B | 1-epoch (run 10-03) | 0.520† | 0.350 | 0.375† | 0.089 | 0.373† | 0.284† |
| Llama 3.1 8B | ep1 | 0.525† | 0.354 | 0.372† | 0.087 | 0.358† | 0.269† |
| Llama 3.1 8B | ep2 | 0.496† | 0.412 | 0.371† | 0.093 | 0.338† | 0.245† |
| Llama 3.1 8B | ep3 | 0.514† | 0.396 | 0.371† | 0.077 | 0.334† | 0.243† |

Best value per column in bold (lowest for hallucination / contradiction). † = differs from the same family's base, paired bootstrap, Holm-adjusted p < 0.05. Judge acc = (correct + 0.5 partial) / N.


**Table 3. Main results, Held-out documents (n = 500 per system)**

| Model | Variant | Judge acc | Halluc. | KF recall | Contra. | F1 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.642 | 0.366 | 0.518 | 0.088 | 0.259 | 0.175 |
| Qwen3-8B | 1-epoch (run 10-03) | 0.485† | 0.365 | 0.341† | 0.106 | 0.355† | 0.267† |
| Qwen3-8B | ep1 | 0.506† | 0.356 | 0.345† | 0.101 | 0.340† | 0.256† |
| Qwen3-8B | ep2 | 0.477† | 0.400 | 0.323† | 0.085 | 0.319† | 0.233† |
| Qwen3-8B | ep3 | 0.476† | 0.417 | 0.337† | 0.091 | 0.315† | 0.227† |
| Gemma 4 E4B | base | **0.679** | **0.218** | **0.541** | **0.065** | 0.177 | 0.118 |
| Gemma 4 E4B | 1-epoch (run 10-03) | 0.518† | 0.312† | 0.357† | 0.083 | **0.361†** | **0.276†** |
| Gemma 4 E4B | ep1 | 0.500† | 0.354† | 0.322† | 0.096 | 0.347† | 0.267† |
| Gemma 4 E4B | ep2 | 0.498† | 0.358† | 0.327† | 0.093 | 0.332† | 0.246† |
| Gemma 4 E4B | ep3 | 0.484† | 0.360† | 0.335† | 0.085 | 0.326† | 0.239† |
| Llama 3.1 8B | base | 0.534 | 0.483 | 0.493 | 0.100 | 0.179 | 0.127 |
| Llama 3.1 8B | 1-epoch (run 10-03) | 0.485† | 0.364† | 0.333† | 0.102 | 0.346† | 0.259† |
| Llama 3.1 8B | ep1 | 0.460† | 0.420 | 0.325† | 0.093 | 0.334† | 0.248† |
| Llama 3.1 8B | ep2 | 0.465† | 0.436 | 0.311† | 0.102 | 0.310† | 0.222† |
| Llama 3.1 8B | ep3 | 0.462† | 0.424 | 0.308† | 0.090 | 0.311† | 0.226† |

Best value per column in bold (lowest for hallucination / contradiction). † = differs from the same family's base, paired bootstrap, Holm-adjusted p < 0.05. Judge acc = (correct + 0.5 partial) / N.


**Table 3. Main results, Seen facts (paraphrased) (n = 279 per system)**

| Model | Variant | Judge acc | Halluc. | KF recall | Contra. | F1 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.649 | 0.297 | 0.523 | 0.108 | 0.213 | 0.141 |
| Qwen3-8B | 1-epoch (run 10-03) | 0.561† | 0.262 | 0.352† | 0.092 | 0.326† | 0.240† |
| Qwen3-8B | ep1 | 0.573† | 0.287 | 0.338† | 0.086 | 0.336† | 0.255† |
| Qwen3-8B | ep2 | 0.590† | 0.323 | 0.426† | 0.083 | 0.366† | 0.283† |
| Qwen3-8B | ep3 | 0.584† | 0.341 | 0.407† | 0.078 | 0.357† | 0.279† |
| Gemma 4 E4B | base | 0.668 | **0.168** | **0.530** | 0.063 | 0.151 | 0.101 |
| Gemma 4 E4B | 1-epoch (run 10-03) | 0.584† | 0.233 | 0.350† | 0.080 | 0.316† | 0.236† |
| Gemma 4 E4B | ep1 | 0.595† | 0.262 | 0.359† | 0.085 | 0.340† | 0.256† |
| Gemma 4 E4B | ep2 | 0.599† | 0.247 | 0.386† | 0.090 | 0.359† | 0.272† |
| Gemma 4 E4B | ep3 | 0.629 | 0.269 | 0.402† | 0.086 | 0.368† | 0.279† |
| Llama 3.1 8B | base | 0.522 | 0.448 | 0.498 | 0.083 | 0.154 | 0.107 |
| Llama 3.1 8B | 1-epoch (run 10-03) | 0.595† | 0.255† | 0.358† | 0.099 | 0.342† | 0.260† |
| Llama 3.1 8B | ep1 | 0.615† | 0.287† | 0.387† | 0.078 | 0.363† | 0.281† |
| Llama 3.1 8B | ep2 | **0.674†** | 0.269† | 0.487 | 0.065 | 0.463† | 0.397† |
| Llama 3.1 8B | ep3 | 0.654† | 0.308† | 0.485 | **0.058** | **0.473†** | **0.411†** |

Best value per column in bold (lowest for hallucination / contradiction). † = differs from the same family's base, paired bootstrap, Holm-adjusted p < 0.05. Judge acc = (correct + 0.5 partial) / N.
