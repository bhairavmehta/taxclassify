# Model card · clf-3b065a184604

| Item | Value |
|---|---|
| Purpose | Classify NTSB reports into the buckets of one grounded taxonomy version (OBJ-3, stage 1) |
| Model | Per-bucket logistic regression (liblinear, C=1.0, balanced class weights) within each CICTT parent; Platt-calibrated on 3-fold out-of-fold scores |
| Features | TF-IDF words and word pairs over narrative + findings (1,598 terms); NTSB finding-code prefixes, phase, aircraft category, CICTT codes (32 tokens, weight 0.5) |
| Abstain | Split-conformal prediction sets, alpha 0.1: a bucket is in the set when p ≥ 0.413; assign only when the set has one bucket and p ≥ 0.5 |
| Training data | 662 rows from run run-combined-20261109T200055880 (hash a75aa638d32a): grounding support links as positives, sibling-moved and sibling support as hard negatives, 8 sampled same-parent negatives per positive; 0 reports excluded |
| Buckets | 27 trained, 3 skipped (fewer than 3 positives) |
| Out-of-fold quality (silver) | mean average precision 0.933 over 27 buckets; per bucket in cv_metrics.csv |
| Known limits | Labels are silver: they come from the retrieval that grounded the buckets, so these scores flatter the model. Unlabelled reports are treated as negatives (PU bias). Conformal coverage is a target until gold labels exist. Evaluate on SME gold labels before use. |
| Reproducibility | Fixed seed 7; model id = hash of training rows, settings, code (96cc8f7396c9) and the run's output hash |
