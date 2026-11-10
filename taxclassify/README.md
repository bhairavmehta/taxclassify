# taxclassify: report classifier for a grounded taxonomy (OBJ-3, stage 1)

Classifies NTSB reports into the buckets of a grounded taxonomy version. It is separate code from `taxground`: it
reads a taxground run folder (or a release's `taxonomy.json`) and needs no database.

## Run it

Everything the classifier needs is in `classifier_input/` (see its README): the NTSB grounding run, 200 held-out
NTSB test reports, a gold-label template, and the synthetic run with its 8 hand-labelled reports.

```bash
pip install -r requirements.txt           # numpy, scikit-learn, scipy, joblib, pytest
./run_classifier.sh                       # 1 tests · 2 synthetic check · 3 train · 4 predict · 5 evaluate (if GOLD) · 6 demo page
open out_classify/<timestamp>/classifier_demo.html

# or step by step
python3 -m taxclassify train    --run classifier_input/ntsb_run --exclude classifier_input/ntsb_test_reports.json
python3 -m taxclassify predict  --model models/clf-<id> --reports classifier_input/ntsb_test_reports.json --out out/predict
python3 -m taxclassify evaluate --model models/clf-<id> --reports reports.json --gold gold.csv --out out/evaluate
python3 -m taxclassify report   --model models/clf-<id> --reports classifier_input/ntsb_test_reports.json \
        --predict out/predict [--eval out/evaluate --gold gold.csv] --out classifier_demo.html
```

| Option | Meaning |
|---|---|
| `train --taxonomy release-v1.0/taxonomy.json` | classify into a release's buckets (default: the run's grounded buckets) |
| `train --include-review` | also train needs-review buckets |
| `train --exclude file ...` | reports never used in training: hold-out and gold files (JSON or CSV with `report_key`) |
| `train --set C=0.5 alpha=0.05 min_prob=0.4 neg_per_pos=8 min_positives=4` | settings (defaults in `model.DEFAULTS`) |
| `evaluate --gold` | CSV `report_key,bucket` (one row per label; empty or `NONE` = none of these), or a JSON with `_expected_bucket` |

## How it works

1. **Training labels (silver), with lineage.** From the grounding run: each bucket's support links are positives;
   reports the sibling rule gave to another bucket, and siblings' support, are hard negatives; 8 random reports per
   positive from the same CICTT parent that no bucket cites are sampled negatives. Every row is written to
   `training_rows.csv` with its origin and its source locator in `grounding_results.json`.
2. **Features.** TF-IDF over words and word pairs of narrative + findings, plus NTSB structure as tokens: the first
   two and three levels of every coded finding with its role (cause, factor, finding), the flight phase, the
   aircraft category and the CICTT codes.
3. **Model.** One logistic regression per bucket (liblinear, balanced class weights, fixed seed), trained only
   against reports under the same parent, so a report is compared with siblings. Platt-calibrated on 3-fold
   out-of-fold scores.
4. **Abstain with conformal sets.** Split-conformal on the out-of-fold probability of each positive's own bucket:
   a bucket enters a report's prediction set when p ≥ 1 − q̂ (alpha 0.10 → a 90 % target). One bucket in the set and
   p ≥ 0.5 → **assign**; two or more → **review**; none → **none_of_these**; no trained bucket under the report's
   codes → **out_of_scope**. Every non-assignment is written as a revision prompt.
5. **Evidence and explanation.** The report sentence the bucket's own model scores highest, quoted verbatim, and
   the five features that pushed the report toward the bucket (coefficient × value).
6. **Checks.** C1 bucket exists in the version · C2 bucket sits under one of the report's codes · C3 evidence is
   verbatim · C4 probabilities in [0, 1] · C5 every non-assignment has a reason. Five injected faults per run must
   all be blocked (`fault_tests.csv`).

## Outputs

| File | Contents |
|---|---|
| `models/clf-<id>/model.joblib` | featurizer + one model per bucket + conformal threshold; `<id>` = hash of training rows, settings, code and the run's output hash |
| `models/clf-<id>/model_card.md` | purpose, model, features, abstain rule, training data, quality, limits, reproducibility |
| `models/clf-<id>/model.json` | the same as data: run id, hashes, sources (sha256), buckets, settings, threshold |
| `models/clf-<id>/training_rows.csv` | every training row with its origin and source |
| `models/clf-<id>/cv_metrics.csv` | per bucket: positives, negatives, out-of-fold average precision and Brier score |
| `<out>/classifications.csv` | per report: decision, label, primary bucket and probability, prediction set, reason, evidence, top features, C1–C5 |
| `<out>/revision_prompts.csv` | non-assignments in taxground's SME decision format, with the prompting report |
| `<out>/fault_tests.csv`, `summary.json` | fault tests; decision counts and check failures |
| `classifier_demo.html` (report) | the walkthrough page: flow and block diagrams, labels, features, PR curves, reliability, conformal histogram, decisions, worked example, checks, evaluation, every prediction |
| `<out>/metrics.json`, `per_bucket.csv` (evaluate) | top-1 accuracy, assigned precision, coverage, conformal set coverage and size, macro-F1, calibration error; per-bucket P/R/F1 |

## First results (10 November 2026)

| Data | Result |
|---|---|
| Synthetic combined run → 8 hand-labelled test reports | 7 of 8 top-1 correct; 5 assigned, all correct; 2 abstained, 1 out of scope; ECE 0.055; 5/5 faults blocked |
| NTSB run 45231e0fc8a9 → 200 recent reports held out of training | 1 assign (correct: bounced landing), 23 review, 140 none of these, 36 out of scope; 0 check failures; 5/5 faults blocked |
| NTSB out-of-fold, silver labels | mean average precision 0.52 over 25 buckets: 0.96 balloon landings, 0.95 helicopter LTE, 0.89 rod/crank failure … 0.24 unreliable airspeed, 0.20 taxi collisions |

With 12 positives per bucket the model is cautious: calibrated probabilities are low, so most reports abstain. That is
the intended failure mode; precision on the gold set decides when to loosen `min_prob`.

## Limits

- **Silver labels.** Training labels come from the retrieval that grounded the buckets, so out-of-fold scores
  flatter the model. Only SME gold labels on a true hold-out (`HOLDOUT_FROM` in `build_all.sh`) measure it.
- **PU bias.** Unlabelled reports under a parent are treated as negatives; some are really positives.
- **Conformal coverage is a target**, computed on silver labels, until a gold calibration set exists.
- **The NTSB test reports above were also in the grounding corpus** (excluded from classifier training only).
- **Next stages** (see the plan doc): SetFit on the approved embedder once SMEs have labelled about 30 reports per
  bucket; an LLM reviewer for abstained reports (DG-3); a fine-tuned encoder when labels reach the hundreds.
