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
