**Table 3. Main results, Seen facts (paraphrased) (n = 279 per system)**

| Model | Variant | Judge acc | Halluc. | KF recall | Contra. | F1 | ROUGE-L |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.649 | 0.297 | – | – | 0.213 | 0.141 |
| Qwen3-8B | 1-epoch (run 10-03) | 0.561† | 0.262 | – | – | 0.326† | 0.240† |
| Qwen3-8B | ep1 | 0.573† | 0.287 | – | – | 0.336† | 0.255† |
| Qwen3-8B | ep2 | 0.590† | 0.323 | – | – | 0.366† | 0.283† |
| Qwen3-8B | ep3 | 0.584† | 0.341 | – | – | 0.357† | 0.279† |
| Gemma 4 E4B | base | 0.668 | **0.168** | – | – | 0.151 | 0.101 |
| Gemma 4 E4B | 1-epoch (run 10-03) | 0.584† | 0.233 | – | – | 0.316† | 0.236† |
| Gemma 4 E4B | ep1 | 0.595† | 0.262 | – | – | 0.340† | 0.256† |
| Gemma 4 E4B | ep2 | 0.599† | 0.247 | – | – | 0.359† | 0.272† |
| Gemma 4 E4B | ep3 | 0.629 | 0.269 | – | – | 0.368† | 0.279† |
| Llama 3.1 8B | base | 0.522 | 0.448 | – | – | 0.154 | 0.107 |
| Llama 3.1 8B | 1-epoch (run 10-03) | 0.595† | 0.255† | – | – | 0.342† | 0.260† |
| Llama 3.1 8B | ep1 | 0.615† | 0.287† | – | – | 0.363† | 0.281† |
| Llama 3.1 8B | ep2 | **0.674†** | 0.269† | – | – | 0.463† | 0.397† |
| Llama 3.1 8B | ep3 | 0.654† | 0.308† | – | – | **0.473†** | **0.411†** |

Best value per column in bold (lowest for hallucination / contradiction). † = differs from the same family's base, paired bootstrap, Holm-adjusted p < 0.05. Judge acc = (correct + 0.5 partial) / N.
