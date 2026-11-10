"""taxclassify command line: train, predict, evaluate.

  python3 -m taxclassify train    --run out_ntsb/latest [--taxonomy out_ntsb/release-v1.0/taxonomy.json] [--exclude holdout.json]
  python3 -m taxclassify predict  --model models/<id> --reports new_reports.json
  python3 -m taxclassify evaluate --model models/<id> --reports reports.json --gold gold.csv
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import sys

import joblib
import numpy as np

from . import __version__
from .checks import CHECKS, check, fault_tests
from .data import Run, gold_labels, load_reports, sha, silver_labels
from .features import Featurizer, text_of
from .model import DEFAULTS, BucketModel, conformal_threshold, evidence

CODE = pathlib.Path(__file__).resolve().parent


def code_hash():
    h = hashlib.sha256()
    for p in sorted(CODE.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- train
def cmd_train(a):
    S = dict(DEFAULTS)
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        S[k] = type(DEFAULTS[k])(v) if k in DEFAULTS else v
    run = Run(a.run)
    statuses = ("grounded", "needs_review") if a.include_review else ("grounded",)
    buckets = run.buckets(a.taxonomy, statuses)
    exclude = set()
    for f in a.exclude or []:
        exclude |= {r["report_key"] for r in load_reports(f)} if f.endswith(".json") else {row["report_key"] for row in csv.DictReader(open(f))}
    rows = silver_labels(run, buckets, S["neg_per_pos"], S["seed"], exclude)
    corpus = [r for k, r in sorted(run.reports.items()) if k not in exclude]
    print(f"run {run.run_id}: {len(buckets)} buckets, {len(rows):,} training rows, featurizer on {len(corpus):,} reports")
    fz = Featurizer(S["max_features"], 2, S["struct_weight"]).fit(corpus)
    keys = sorted({r["report_key"] for r in rows})
    X = fz.transform([run.reports[k] for k in keys])
    at = {k: i for i, k in enumerate(keys)}
    models, skipped = {}, {}
    for b, info in sorted(buckets.items()):
        br = [r for r in rows if r["bucket"] == b]
        y = np.array([r["y"] for r in br])
        npos, nneg = int(y.sum()), int(len(y) - y.sum())
        if npos < S["min_positives"] or nneg < 2:
            skipped[b] = f"{npos} positives, {nneg} negatives (needs {S['min_positives']} and 2)"
            continue
        Xi = X[[at[r["report_key"]] for r in br]]
        models[b] = BucketModel(b, info["parent"]).fit(Xi, y, np.array([r["weight"] for r in br]), S)
    qhat, ncal = conformal_threshold(models, S["alpha"])
    # model id = hash of training rows, settings, code and the run's inputs
    h = hashlib.sha256()
    h.update(json.dumps(rows, sort_keys=True).encode()); h.update(json.dumps(S, sort_keys=True).encode())
    h.update(code_hash().encode()); h.update(run.summary.get("output_hash", "").encode())
    mid = h.hexdigest()[:12]
    out = pathlib.Path(a.out) / f"clf-{mid}"
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(model_id=mid, built=now(), taxclassify_version=__version__, code_hash=code_hash(), run_id=run.run_id,
                run_output_hash=run.summary.get("output_hash"), taxonomy=str(a.taxonomy or run.taxonomy_path),
                settings=S, buckets=buckets, trained=sorted(models), skipped=skipped, conformal=dict(alpha=S["alpha"], qhat=qhat,
                threshold=1 - qhat, calibration_positives=ncal, labels="silver (grounding support links)"),
                sources={str(run.results_path): sha(run.results_path), str(run.reports_path): sha(run.reports_path)},
                excluded_reports=len(exclude))
    joblib.dump(dict(meta=meta, featurizer=fz, models=models), out / "model.joblib", compress=3)
    (out / "model.json").write_text(json.dumps(meta, indent=1, default=str))
    with open(out / "training_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["bucket", "report_key", "y", "weight", "origin", "source"]); w.writeheader(); w.writerows(rows)
    with open(out / "cv_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bucket", "parent", "positives", "negatives", "oof_average_precision", "oof_brier"])
        for b, m in sorted(models.items()):
            w.writerow([b, m.parent, m.n_pos, m.n_neg, round(m.ap, 4), round(m.brier, 4)])
    aps = [m.ap for m in models.values() if not np.isnan(m.ap)]
    card = f"""# Model card · clf-{mid}

| Item | Value |
|---|---|
| Purpose | Classify NTSB reports into the buckets of one grounded taxonomy version (OBJ-3, stage 1) |
| Model | Per-bucket logistic regression (liblinear, C={S['C']}, balanced class weights) within each CICTT parent; Platt-calibrated on {S['folds']}-fold out-of-fold scores |
| Features | TF-IDF words and word pairs over narrative + findings ({len(fz.text.vocabulary_):,} terms); NTSB finding-code prefixes, phase, aircraft category, CICTT codes ({len(fz.struct.vocabulary_):,} tokens, weight {S['struct_weight']}) |
| Abstain | Split-conformal prediction sets, alpha {S['alpha']}: a bucket is in the set when p ≥ {1 - qhat:.3f}; assign only when the set has one bucket and p ≥ {S['min_prob']} |
| Training data | {len(rows):,} rows from run {run.run_id} (hash {run.summary.get('output_hash', '')[:12]}): grounding support links as positives, sibling-moved and sibling support as hard negatives, {S['neg_per_pos']} sampled same-parent negatives per positive; {len(exclude):,} reports excluded |
| Buckets | {len(models)} trained, {len(skipped)} skipped (fewer than {S['min_positives']} positives) |
| Out-of-fold quality (silver) | mean average precision {np.mean(aps) if aps else float('nan'):.3f} over {len(aps)} buckets; per bucket in cv_metrics.csv |
| Known limits | Labels are silver: they come from the retrieval that grounded the buckets, so these scores flatter the model. Unlabelled reports are treated as negatives (PU bias). Conformal coverage is a target until gold labels exist. Evaluate on SME gold labels before use. |
| Reproducibility | Fixed seed {S['seed']}; model id = hash of training rows, settings, code ({code_hash()}) and the run's output hash |
"""
    (out / "model_card.md").write_text(card)
    print(card)
    print(f"→ {out}")


# ---------------------------------------------------------------- predict
def classify(bundle, reports):
    meta, fz, models = bundle["meta"], bundle["featurizer"], bundle["models"]
    S, buckets, thr = meta["settings"], meta["buckets"], meta["conformal"]["threshold"]
    X = fz.transform(reports)
    rows = []
    for i, r in enumerate(reports):
        codes = r.get("l1_codes", [])
        cands = [b for b, m in models.items() if m.parent in codes]
        row = dict(report_key=r["report_key"], codes=" ".join(codes), labels=[], primary="", primary_prob=None, scores=[],
                   prediction_set=[], decision="", reason="", evidence="", top_features=[])
        if not cands:
            row.update(decision="out_of_scope", reason=f"no trained bucket under {', '.join(codes) or 'no code'}: consider a new bucket")
            rows.append(row)
            continue
        xi = X[i]
        sc = sorted(((b, float(models[b].prob(xi)[0])) for b in cands), key=lambda t: -t[1])
        best, p = sc[0]
        pset = [(b, q) for b, q in sc if q >= thr]
        row.update(scores=[(b, round(q, 4)) for b, q in sc], primary=best, primary_prob=round(p, 4),
                   prediction_set=[(b, round(q, 4)) for b, q in pset],
                   evidence=evidence(models[best], fz, text_of(r)), top_features=models[best].contributions(xi, fz.names))
        if not pset:
            row.update(decision="none_of_these", reason=f"no bucket reaches the conformal threshold {thr:.2f}; nearest {best} at {p:.2f}: new bucket or widen {best}")
        elif len(pset) > 1:
            row.update(decision="review", reason=f"{len(pset)} buckets in the {1 - S['alpha']:.0%} set ({', '.join(b for b, _ in pset)}): SME picks one or both; sharpen exclusions")
        elif p < S["min_prob"]:
            row.update(decision="review", reason=f"only {best} in the set but p = {p:.2f} < {S['min_prob']}: SME confirms")
        else:
            row.update(decision="assign", labels=[best])
        rows.append(row)
    return rows


def write_predictions(rows, reports, bundle, out):
    meta, buckets = bundle["meta"], bundle["meta"]["buckets"]
    out.mkdir(parents=True, exist_ok=True)
    rep = {r["report_key"]: r for r in reports}
    with open(out / "classifications.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["report_key", "cictt_codes", "decision", "label", "primary", "primary_prob", "prediction_set", "reason",
                    "evidence", "top_features", "model_id", "taxonomy_run"] + [c for c, _ in CHECKS])
        allchecks = {}
        for row in rows:
            r = rep[row["report_key"]]
            res = check(row, buckets, text_of(r), r.get("l1_codes", []))
            allchecks[row["report_key"]] = res
            w.writerow([row["report_key"], row["codes"], row["decision"], ";".join(row["labels"]), row["primary"], row["primary_prob"],
                        "; ".join(f"{b}:{p:.2f}" for b, p in row["prediction_set"]), row["reason"], row["evidence"],
                        "; ".join(f"{n} ({c:+.3f})" for n, c in row["top_features"]), meta["model_id"], meta["run_id"]]
                       + ["pass" if ok else "FAIL" for _, ok in res])
    with open(out / "revision_prompts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reviewed_run_id", "candidate_id", "action", "target_type", "target_id", "value", "rationale", "prompting_report", "suggestion"])
        for row in rows:
            if row["decision"] != "assign":
                w.writerow([meta["run_id"], row["primary"], "", "", "", "", "", row["report_key"], f"classifier {row['decision']}: {row['reason']}"])
    ok_row = next((x for x in rows if x["decision"] == "assign"), None)
    faults = fault_tests(ok_row, buckets, text_of(rep[ok_row["report_key"]]), rep[ok_row["report_key"]].get("l1_codes", [])) if ok_row else []
    with open(out / "fault_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["test", "check", "blocked"]); w.writeheader(); w.writerows(faults)
    failed = sum(1 for v in allchecks.values() for _, ok in v if not ok)
    dec = {}
    for x in rows:
        dec[x["decision"]] = dec.get(x["decision"], 0) + 1
    summ = dict(model_id=meta["model_id"], reports=len(rows), decisions=dec, check_failures=failed,
                faults_blocked=f"{sum(f['blocked'] for f in faults)}/{len(faults)}", written=now())
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def cmd_predict(a):
    bundle = joblib.load(pathlib.Path(a.model) / "model.joblib")
    reports = load_reports(a.reports)
    rows = classify(bundle, reports)
    out = pathlib.Path(a.out or f"out_classify/classify-{dt.datetime.now():%Y%m%dT%H%M%S}-{bundle['meta']['model_id']}")
    summ = write_predictions(rows, reports, bundle, out)
    print(json.dumps(summ, indent=1)); print(f"→ {out}")
    return 1 if summ["check_failures"] else 0


# ---------------------------------------------------------------- evaluate
def ece(probs, correct, bins=10):
    probs, correct = np.asarray(probs, float), np.asarray(correct, float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (probs >= lo) & (probs < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(probs[m].mean() - correct[m].mean())
    return float(e)


def cmd_evaluate(a):
    bundle = joblib.load(pathlib.Path(a.model) / "model.joblib")
    gold = gold_labels(a.gold)
    reports = [r for r in load_reports(a.reports) if r["report_key"] in gold]
    if not reports:
        sys.exit("no report in --reports has a gold label")
    trained = set(bundle["models"])
    rows = classify(bundle, reports)
    out = pathlib.Path(a.out or f"out_classify/eval-{dt.datetime.now():%Y%m%dT%H%M%S}-{bundle['meta']['model_id']}")
    write_predictions(rows, reports, bundle, out)
    per = {b: dict(tp=0, fp=0, fn=0) for b in sorted(trained)}
    top1, cover, sizes, probs, correct, assigned, assigned_ok = 0, 0, [], [], [], 0, 0
    for x in rows:
        g = gold[x["report_key"]] & trained
        none_gold = not gold[x["report_key"]]
        hit = (x["primary"] in g) if g else (none_gold and x["decision"] in ("none_of_these", "out_of_scope"))
        top1 += hit
        if g:
            cover += bool(g & {b for b, _ in x["prediction_set"]})
        sizes.append(len(x["prediction_set"]))
        if x["primary_prob"] is not None:
            probs.append(x["primary_prob"]); correct.append(x["primary"] in g)
        if x["decision"] == "assign":
            assigned += 1; assigned_ok += x["primary"] in g
            for b in x["labels"]:
                per[b]["tp" if b in g else "fp"] += 1
            for b in g - set(x["labels"]):
                per[b]["fn"] += 1
        else:
            for b in g:
                per[b]["fn"] += 1
    rowsb = []
    for b, c in per.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
        f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
        rowsb.append(dict(bucket=b, **c, precision=round(p, 3) if p == p else "", recall=round(r, 3) if r == r else "", f1=round(f1, 3) if f1 == f1 else ""))
    with open(out / "per_bucket.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["bucket", "tp", "fp", "fn", "precision", "recall", "f1"]); w.writeheader(); w.writerows(rowsb)
    f1s = [x["f1"] for x in rowsb if x["f1"] != ""]
    n_g = sum(1 for x in rows if gold[x["report_key"]] & trained)
    m = dict(model_id=bundle["meta"]["model_id"], gold_file=str(a.gold), reports=len(rows), top1_accuracy=round(top1 / len(rows), 3),
             assigned=assigned, assigned_precision=round(assigned_ok / assigned, 3) if assigned else None,
             coverage=round(assigned / len(rows), 3), set_coverage=round(cover / n_g, 3) if n_g else None,
             conformal_target=1 - bundle["meta"]["settings"]["alpha"], mean_set_size=round(float(np.mean(sizes)), 2),
             macro_f1=round(float(np.mean(f1s)), 3) if f1s else None, ece=round(ece(probs, correct), 3) if probs else None)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1)); print(f"→ {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="taxclassify", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a classifier from a grounding run")
    t.add_argument("--run", required=True, help="taxground run folder (e.g. out_ntsb/latest)")
    t.add_argument("--taxonomy", help="a release taxonomy.json: classify into its buckets (default: the run's grounded buckets)")
    t.add_argument("--include-review", action="store_true", help="also train needs-review buckets (no release given)")
    t.add_argument("--exclude", nargs="*", help="reports never used for training (hold-out, gold): JSON or CSV with report_key")
    t.add_argument("--set", nargs="*", help=f"override settings, e.g. C=0.5 alpha=0.05 ({', '.join(DEFAULTS)})")
    t.add_argument("--out", default="models")
    p = sub.add_parser("predict", help="classify reports")
    p.add_argument("--model", required=True); p.add_argument("--reports", required=True); p.add_argument("--out")
    e = sub.add_parser("evaluate", help="score predictions against gold labels")
    e.add_argument("--model", required=True); e.add_argument("--reports", required=True)
    e.add_argument("--gold", required=True, help="CSV report_key,bucket (empty or NONE = none of these) or a JSON with _expected_bucket")
    e.add_argument("--out")
    r = sub.add_parser("report", help="build the self-contained demo page from a model, its predictions and an evaluation")
    r.add_argument("--model", required=True); r.add_argument("--reports", required=True, help="the reports that were classified")
    r.add_argument("--predict", help="predict output folder (fault tests)"); r.add_argument("--eval", help="evaluate output folder")
    r.add_argument("--gold", help="gold labels used by --eval"); r.add_argument("--out", default="classifier_demo.html")
    r.add_argument("--json", help="also write the page's data as JSON")
    a = ap.parse_args(argv)
    return {"train": cmd_train, "predict": cmd_predict, "evaluate": cmd_evaluate, "report": cmd_report}[a.cmd](a) or 0


def cmd_report(a):
    from .report import build, write
    data = build(a.model, a.reports, a.predict, a.eval, a.gold)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(data, indent=1, default=float))
    write(data, a.out)
    print(f"→ {a.out} ({pathlib.Path(a.out).stat().st_size / 1e6:.2f} MB)")
