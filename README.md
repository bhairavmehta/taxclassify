# taxclassify package: classify NTSB reports into grounded buckets

Self-contained: code, input data, a trained model and a finished demo run. No database needed.

| Folder / file | What it is |
|---|---|
| `taxclassify/` | classifier code (data, features, model, checks, cli, report + demo_template.html); see its README |
| `run_classifier.sh` | end to end: tests → synthetic answer-key check → train → predict → evaluate → demo page |
| `classifier_input/` | inputs: NTSB grounding run (run-ntsb-20261110T185650894), 200 held-out test reports, gold template, synthetic run + 8 labelled reports |
| `tests/test_classify.py` | unit tests, including train → evaluate on the synthetic run |
| `requirements.txt` | numpy, scipy, scikit-learn, joblib, pytest |
| `models/clf-ae93f2c4a6d5/` | the trained model from the demo run: model.joblib, model_card.md, training_rows.csv, cv_metrics.csv |
| `out_classify/demo/` | the demo run: predict/, synthetic_eval/, classifier_demo.html |

## Execution steps

```bash
unzip taxclassify_package.zip && cd taxclassify_package
pip install -r requirements.txt          # Python 3.10+
./run_classifier.sh                      # about 35 s; writes models/clf-<id> and out_classify/<timestamp>/
# open out_classify/<timestamp>/classifier_demo.html in a browser
```

Once SMEs have labelled reports in `classifier_input/ntsb_gold_labels_TEMPLATE.csv` (save as `ntsb_gold_labels.csv`):

```bash
GOLD=classifier_input/ntsb_gold_labels.csv ./run_classifier.sh
```

The same inputs and code give the same model id (`clf-ae93f2c4a6d5`).
