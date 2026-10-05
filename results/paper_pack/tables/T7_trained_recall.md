**Table 7. Trained-question recall: the exact training questions (n = 500)**

| Model | Variant | Judge acc | KF recall | Contra. | F1 | Exact repro. | ROUGE-L | Words |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | base | – | – | – | 0.266 | 0.000 | 0.186 | 123.9 |
| Qwen3-8B | concise base (<= 40 words) | – | – | – | 0.313† | 0.004 | 0.238† | 24.6† |
| Qwen3-8B | 1-epoch (run 10-03) | – | – | – | 0.409† | 0.016 | 0.325† | 33.6† |
| Qwen3-8B | ep1 | – | – | – | 0.445† | 0.054† | 0.368† | 32.5† |
| Qwen3-8B | ep2 | – | – | – | 0.537† | 0.184† | 0.474† | 38.7† |
| Qwen3-8B | ep3 | – | – | – | **0.561†** | **0.214†** | **0.502†** | 38.1† |

Judge acc = (correct + 0.5 partial) / N (local judge, prompt v1). KF recall = share of reference key facts supported by the answer (MiniCheck-7B). Exact repro. = share of answers with ROUGE-L ≥ 0.8 against the trained answer. † = differs from the same model's base (paired bootstrap, Holm-adjusted p < 0.05). Best per column in bold.
