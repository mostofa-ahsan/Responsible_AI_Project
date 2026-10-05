**Table 4. Judge and checker validation against Opus 5.5**

| Check | Statistic | Value |
|---|---|---|
| Local judge vs Opus (TEST, n = 6673) | 3-class agreement | 0.646 |
|  | Cohen's κ / quadratic-weighted κ | 0.453 / 0.644 |
|  | binary κ correct-vs-rest / incorrect-vs-rest | 0.482 / 0.606 |
|  | system-level Pearson r / Spearman ρ (18 cells) | 0.900 / 0.806 |
|  | base→FT delta signs matching | 8/9 |
| Key-fact recall vs Opus (item level) | AUC key-fact recall → Opus correct | 0.852 |
| Key-fact recall vs Opus (item level) | AUC key-fact recall (secondary) → Opus correct | 0.823 |
| Key-fact recall vs Opus (item level) | AUC key-fact recall → Opus not incorrect | 0.834 |
| Contradiction vs Opus (item level) | r_pb any contradiction ~ Opus incorrect | 0.320 (n = 7674) |
| Contradiction vs Opus (item level) | r_pb contradiction rate ~ Opus incorrect | 0.351 (n = 7674) |
| Contradiction vs Opus (item level) | r_pb contradicted-claim rate (2c) ~ Opus incorrect | 0.272 (n = 6395) |
| Contradiction vs Opus (item level) | r_pb judge hallucination flag ~ Opus incorrect | 0.387 (n = 7667) |
| System level vs Opus accuracy | Pearson / Spearman: key-fact recall | 0.925 / 0.988 |
| System level vs Opus accuracy | Pearson / Spearman: contradiction rate | -0.303 / -0.321 |
| System level vs Opus accuracy | Pearson / Spearman: local-judge accuracy (all items) | 0.888 / 0.800 |
| System level vs Opus accuracy | Pearson / Spearman: token F1 | -0.701 / -0.670 |
| System level vs Opus accuracy | Pearson / Spearman: ROUGE-L | -0.728 / -0.659 |
| MiniCheck-7B vs Flan-T5 (n = 67500 facts) | agreement / κ / prob. Pearson | 0.849 / 0.684 / 0.801 |
|  | support rate MiniCheck-7B / Flan-T5 | 0.392 / 0.398 |

Opus grades exist for the 3 base and 3 one-epoch systems only (eval subset).
