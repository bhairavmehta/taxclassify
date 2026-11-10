# Classifier input data

Everything `taxclassify` reads, copied here so the classifier runs without the grounding database or a rebuild.

| Path | What it is | Where it came from |
|---|---|---|
| `ntsb_run/grounding_results.json` | the grounding run: buckets, status, parent, support links (training positives), sibling moves (hard negatives) | `out_ntsb/run-ntsb-20261110T192033918` (output hash 45231e0fc8a9) |
| `ntsb_run/taxonomy.json` | the versioned hierarchy of that run (buckets per CICTT parent) | same run |
| `ntsb_run/manifest.json`, `links.sssom.tsv` | run summary, settings, input hashes; the stored link rows | same run |
| `ntsb_run/inputs/sample_reports.json` | 27,895 NTSB reports 2008–2026: narrative, verbalized findings, CICTT codes, metadata (sha256 d4d63e0f83d0…) | `build_all.sh` steps 1–2 from NTSB CAROL avall.zip |
| `ntsb_run/inputs/*.json` | candidates (53), CICTT parents (33), definitions (548) | steps 2, 3, 6 |
| `ntsb_test_reports.json` | 200 most recent NTSB reports, kept out of classifier training (they were in the grounding corpus) | `build_all.sh` step 4 |
| `ntsb_gold_labels_TEMPLATE.csv` | blank gold-label sheet for those 200 reports: SMEs write the bucket id (or NONE) per report; one row per label | for `taxclassify evaluate --gold` |
| `synthetic_run/` | the synthetic combined run from the trial package (hand-labelled answer key) | `out/run-combined-20261109T200055880` |
| `synthetic_test_reports.json` | 8 synthetic test reports with `_expected_bucket` (the answer key) | `data/test_reports.json` |

To use a newer grounding run or a release, pass `RUN=` / `TAXONOMY=` to `run_classifier.sh`.
