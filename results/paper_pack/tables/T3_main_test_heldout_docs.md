**Table 3. Main results, Held-out documents (n = 500 per system)**

| Model | Variant | Judge acc | Halluc. | KF recall | Contra. | F1 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.642 | 0.366 | – | – | 0.259 | 0.175 |
| Qwen3-8B | 1-epoch (run 10-03) | 0.485† | 0.365 | – | – | 0.355† | 0.267† |
| Qwen3-8B | ep1 | 0.506† | 0.356 | – | – | 0.340† | 0.256† |
| Qwen3-8B | ep2 | 0.477† | 0.400 | – | – | 0.319† | 0.233† |
| Qwen3-8B | ep3 | 0.476† | 0.417 | – | – | 0.315† | 0.227† |
| Gemma 4 E4B | base | **0.679** | **0.218** | – | – | 0.177 | 0.118 |
| Gemma 4 E4B | 1-epoch (run 10-03) | 0.518† | 0.312† | – | – | **0.361†** | **0.276†** |
| Gemma 4 E4B | ep1 | 0.500† | 0.354† | – | – | 0.347† | 0.267† |
| Gemma 4 E4B | ep2 | 0.498† | 0.358† | – | – | 0.332† | 0.246† |
| Gemma 4 E4B | ep3 | 0.484† | 0.360† | – | – | 0.326† | 0.239† |
| Llama 3.1 8B | base | 0.534 | 0.483 | – | – | 0.179 | 0.127 |
| Llama 3.1 8B | 1-epoch (run 10-03) | 0.485† | 0.364† | – | – | 0.346† | 0.259† |
| Llama 3.1 8B | ep1 | 0.460† | 0.420 | – | – | 0.334† | 0.248† |
| Llama 3.1 8B | ep2 | 0.465† | 0.436 | – | – | 0.310† | 0.222† |
| Llama 3.1 8B | ep3 | 0.462† | 0.424 | – | – | 0.311† | 0.226† |

Best value per column in bold (lowest for hallucination / contradiction). † = differs from the same family's base, paired bootstrap, Holm-adjusted p < 0.05. Judge acc = (correct + 0.5 partial) / N.
