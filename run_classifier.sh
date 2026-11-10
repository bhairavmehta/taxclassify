#!/usr/bin/env bash
# taxclassify end to end: tests → synthetic answer-key check → train on the NTSB grounding run → classify →
# (evaluate against SME gold labels) → demo page. Reads classifier_input/ by default; no database needed.
#
#   ./run_classifier.sh
#   GOLD=classifier_input/ntsb_gold_labels.csv ./run_classifier.sh          # once SMEs have filled the template
#   RUN=out_ntsb/latest REPORTS=new_reports.json ./run_classifier.sh          # a newer run, other reports
#   TAXONOMY=out_ntsb/release-v1.0/taxonomy.json ./run_classifier.sh          # classify into a release's buckets
#   STEPS="3 4 6" SETTINGS="min_prob=0.4 alpha=0.05" ./run_classifier.sh
#
# Steps
#   1 tests       unit tests (grounding + classifier)
#   2 synthetic   train on the synthetic run, evaluate on its 8 hand-labelled test reports (answer key)
#   3 train       train on the NTSB run; the reports to classify and any gold labels are excluded from training
#   4 predict     classify the reports: assign / review / none_of_these / out_of_scope, evidence, checks C1–C5, fault tests
#   5 evaluate    against GOLD (SME labels) when given
#   6 report      classifier_demo.html: every step, chart and outcome, from the files above
set -euo pipefail
cd "$(dirname "$0")"
IN=${IN:-classifier_input}
RUN=${RUN:-$IN/ntsb_run}
REPORTS=${REPORTS:-$IN/ntsb_test_reports.json}
GOLD=${GOLD:-}
TAXONOMY=${TAXONOMY:-}
SETTINGS=${SETTINGS:-}
OUT=${OUT:-out_classify/$(date +%Y%m%dT%H%M%S)}
MODELS=${MODELS:-models}
STEPS=${STEPS:-"1 2 3 4 5 6"}
on() { [[ " $STEPS " == *" $1 "* ]]; }
say() { printf '\n\033[1;33m== %s\033[0m\n' "$*"; }
mkdir -p "$OUT"

if on 1; then say "1. unit tests"; python3 -m pytest -q tests; fi

if on 2; then
  say "2. synthetic answer key: train on $IN/synthetic_run, evaluate on 8 labelled reports"
  python3 -m taxclassify train --run "$IN/synthetic_run" --include-review --out "$OUT/synthetic_models" --set min_positives=3 | grep -E "Buckets|Out-of|→"
  SM=$(ls -td "$OUT"/synthetic_models/clf-* | head -1)
  python3 -m taxclassify evaluate --model "$SM" --reports "$IN/synthetic_test_reports.json" --gold "$IN/synthetic_test_reports.json" --out "$OUT/synthetic_eval" | tail -14
fi

if on 3; then
  say "3. train on $RUN"
  EXCL=("$REPORTS"); [ -n "$GOLD" ] && EXCL+=("$GOLD")
  python3 -m taxclassify train --run "$RUN" ${TAXONOMY:+--taxonomy "$TAXONOMY"} --exclude "${EXCL[@]}" --out "$MODELS" ${SETTINGS:+--set $SETTINGS} | tail -12
fi
MODEL=$(ls -td "$MODELS"/clf-* 2>/dev/null | head -1 || true)

if on 4; then say "4. classify $REPORTS"; python3 -m taxclassify predict --model "$MODEL" --reports "$REPORTS" --out "$OUT/predict"; fi

if on 5; then
  if [ -n "$GOLD" ]; then say "5. evaluate against $GOLD"; python3 -m taxclassify evaluate --model "$MODEL" --reports "$REPORTS" --gold "$GOLD" --out "$OUT/evaluate"
  else say "5. evaluate: skipped (no GOLD; fill $IN/ntsb_gold_labels_TEMPLATE.csv and pass GOLD=)"; fi
fi

if on 6; then
  say "6. demo page"
  EV="$OUT/synthetic_eval"; GL="$IN/synthetic_test_reports.json"
  [ -n "$GOLD" ] && [ -d "$OUT/evaluate" ] && { EV="$OUT/evaluate"; GL="$GOLD"; }
  python3 -m taxclassify report --model "$MODEL" --reports "$REPORTS" --predict "$OUT/predict" \
    $([ -d "$EV" ] && echo --eval "$EV" --gold "$GL") --out "$OUT/classifier_demo.html" --json "$OUT/classifier_demo.json"
fi
say "done"
echo "  model:   $MODEL (model_card.md, training_rows.csv, cv_metrics.csv)"
echo "  results: $OUT (predict/, synthetic_eval/, classifier_demo.html)"
