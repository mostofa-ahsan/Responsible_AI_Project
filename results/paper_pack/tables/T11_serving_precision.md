**Table 11. Serving precision: adapter on the bf16 base vs on its training (NF4) base**

| Model | Adapter | Test | Metric | bf16 base | Training base | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|

Same adapters, same pre-tokenized greedy decoding except the base weights: bf16 (earlier results) vs the NF4-dequantized bf16 weights the adapter was trained on. Paired bootstrap on identical items.
