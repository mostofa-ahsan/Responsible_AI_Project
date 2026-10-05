**Table 7. Trained-question recall: the exact training questions (n = 500)**

| Model | Variant | Judge acc | KF recall | Contra. | F1 | Exact repro. | ROUGE-L | Words |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | 0.657 | 0.559 | 0.093 | 0.266 | 0.000 | 0.186 | 123.9 |
| Qwen3-8B | concise base (<= 40 words) | 0.534† | 0.332† | 0.096 | 0.313† | 0.004 | 0.238† | 24.6† |
| Qwen3-8B | 1-epoch (run 10-03) | 0.546† | 0.361† | 0.085 | 0.409† | 0.016 | 0.325† | 33.6† |
| Qwen3-8B | ep1 | 0.593† | 0.399† | 0.069 | 0.445† | 0.054† | 0.368† | 32.5† |
| Qwen3-8B | ep2 | 0.667 | 0.511† | 0.048† | 0.537† | 0.184† | 0.474† | 38.7† |
| Qwen3-8B | ep3 | 0.670 | 0.529 | 0.055 | 0.561† | 0.214† | 0.502† | 38.1† |
| Qwen3-8B | NF4 base + adapter (ep1) | 0.590† | 0.399† | 0.064 | 0.455† | 0.064† | 0.379† | 31.5† |
| Qwen3-8B | merged 4-bit (AWQ/W4A16, vLLM) (ep1) | 0.572† | 0.398† | 0.076 | 0.426† | 0.036† | 0.345† | 34.2† |
| Gemma 4 E4B | base | 0.702 | 0.562 | 0.073 | 0.173 | 0.000 | 0.118 | 227.9 |
| Gemma 4 E4B | concise base (<= 40 words) | 0.548† | 0.322† | 0.078 | 0.290† | 0.002 | 0.212† | 28.0† |
| Gemma 4 E4B | 1-epoch (run 10-03) | 0.547† | 0.349† | 0.080 | 0.405† | 0.014 | 0.319† | 30.7† |
| Gemma 4 E4B | ep1 | 0.563† | 0.365† | 0.073 | 0.422† | 0.026 | 0.341† | 30.8† |
| Gemma 4 E4B | ep2 | 0.614† | 0.422† | 0.068 | 0.467† | 0.066† | 0.394† | 33.6† |
| Gemma 4 E4B | ep3 | 0.638† | 0.432† | 0.061 | 0.486† | 0.090† | 0.413† | 34.0† |
| Gemma 4 E4B | NF4 base + adapter (ep1) | 0.576† | 0.377† | 0.071 | 0.427† | 0.042† | 0.354† | 30.9† |
| Gemma 4 E4B | merged 4-bit (AWQ/W4A16, vLLM) (ep1) | 0.528† | 0.348† | 0.083 | 0.396† | 0.014 | 0.314† | 31.7† |
| Llama 3.1 8B | base | 0.552 | 0.517 | 0.100 | 0.179 | 0.000 | 0.129 | 222.1 |
| Llama 3.1 8B | concise base (<= 40 words) | 0.495† | 0.369† | 0.105 | 0.315† | 0.000 | 0.231† | 37.8† |
| Llama 3.1 8B | 1-epoch (run 10-03) | 0.571 | 0.378† | 0.072 | 0.431† | 0.040† | 0.351† | 33.5† |
| Llama 3.1 8B | ep1 | 0.607† | 0.430† | 0.058† | 0.481† | 0.080† | 0.413† | 35.1† |
| Llama 3.1 8B | ep2 | 0.853† | 0.726† | 0.029† | 0.811† | 0.658† | 0.788† | 38.8† |
| Llama 3.1 8B | ep3 | 0.896† | 0.751† | 0.019† | 0.856† | 0.742† | 0.839† | 38.1† |
| Llama 3.1 8B | NF4 base + adapter (ep2) | **0.962†** | **0.821†** | **0.009†** | **0.943†** | **0.900†** | **0.934†** | 38.2† |
| Llama 3.1 8B | merged 4-bit (AWQ/W4A16, vLLM) (ep2) | 0.687† | 0.573† | 0.045† | 0.651† | 0.342† | 0.609† | 39.6† |

Judge acc = (correct + 0.5 partial) / N (local judge, prompt v1). KF recall = share of reference key facts supported by the answer (MiniCheck-7B). Exact repro. = share of answers with ROUGE-L ≥ 0.8 against the trained answer. † = differs from the same model's base (paired bootstrap, Holm-adjusted p < 0.05). Best per column in bold.
