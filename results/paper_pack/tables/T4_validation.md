**Table 4. Judge and checker validation against Opus 5.5**

| Check | Statistic | Value |
|---|---|---|
| Local judge vs Opus (TEST, n = 6673) | 3-class agreement | 0.646 |
|  | Cohen's κ / quadratic-weighted κ | 0.453 / 0.644 |
|  | binary κ correct-vs-rest / incorrect-vs-rest | 0.482 / 0.606 |
|  | system-level Pearson r / Spearman ρ (18 cells) | 0.900 / 0.806 |
|  | base→FT delta signs matching | 8/9 |
| Contradiction vs Opus (item level) | r_pb judge hallucination flag ~ Opus incorrect | 0.387 (n = 7667) |
| System level vs Opus accuracy | Pearson / Spearman: local-judge accuracy (all items) | 0.888 / 0.800 |
| System level vs Opus accuracy | Pearson / Spearman: token F1 | -0.701 / -0.670 |
| System level vs Opus accuracy | Pearson / Spearman: ROUGE-L | -0.728 / -0.659 |

Opus grades exist for the 3 base and 3 one-epoch systems only (eval subset).
