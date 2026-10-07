# Figure captions

**F2_accuracy_four_tests**: Judge accuracy (correct + 0.5 partial) on the four test types for the base model, the concise base (≤ 40 words), the best QA-only epoch and the best mixed-CPT epoch (adapters served on their training base); 95% bootstrap CIs.

**F3_accuracy_and_val_loss_vs_epoch**: Mean judge accuracy on unseen questions (in-domain + held-out) against epoch (0 = base; coloured, left axis) and validation loss (grey, right axis), QA-only (solid) and mixed CPT (dashed).

**F4_exact_vs_reworded**: Exact training question vs its rewording on the 279 paired items (filled: exact, open: reworded) for base, best QA-only and best mixed-CPT systems.

**F5_length_decomposition**: Held-out documents: change in judge accuracy from shortening the base answer (concise base − base) and from fine-tuning at matched length (best QA-only − concise base).

**F6_metrics_vs_opus**: Token F1 (left) and key-fact recall (right) against Opus 5.5 accuracy for the 18 system × test cells graded by Opus (bf16-served base and 1-epoch systems); Pearson r shown.

**F7_judge_validation**: Local judge vs Opus 5.5 on held-out calibration items: confusion matrix (left) and system × test accuracy (right; circles base, triangles 1-epoch fine-tuned).

**F8_serving_precision_deployment**: Left: the best QA-only adapter served on the bf16 base (open) vs on its NF4 training base (filled). Right: deployment variants that exist (earlier deployment run; merged 4-bit variants were merged into the bf16 base).
