"""Build the classifier demo page: every chart and table is computed here from the saved model, its predictions and
(optionally) an evaluation, and embedded in one self-contained HTML file (taxclassify/demo_template.html).

  python3 -m taxclassify report --model models/clf-<id> --reports build/test_reports.json \
      [--predict out_classify/x/predict] [--eval out_classify/y/evaluate --gold data/test_reports.json] --out classifier_demo.html
"""
import collections
import csv
import json
import pathlib

import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve

from .checks import CHECKS
from .data import gold_labels, load_reports
from .features import structured_tokens, text_of

HERE = pathlib.Path(__file__).resolve().parent


def _kind(name):
    for p, k in (("F:", "finding"), ("PH:", "phase"), ("L1:", "cictt"), ("CAT:", "category")):
        if name.startswith(p):
            return k
    return "text"


def _pr(y, p, n=40):
    prec, rec, _ = precision_recall_curve(y, p)
    idx = np.unique(np.linspace(0, len(prec) - 1, min(n, len(prec))).astype(int))
    return [[round(float(rec[i]), 4), round(float(prec[i]), 4)] for i in idx]


def build(model_dir, reports_path, predict_dir=None, eval_dir=None, gold_path=None):
    from .cli import classify
    bundle = joblib.load(pathlib.Path(model_dir) / "model.joblib")
    meta, fz, models = bundle["meta"], bundle["featurizer"], bundle["models"]
    md = pathlib.Path(model_dir)
    reports = load_reports(reports_path)
    rep = {r["report_key"]: r for r in reports}
    rows = classify(bundle, reports)

    # training composition (from training_rows.csv)
    comp = collections.defaultdict(lambda: collections.Counter())
    for t in csv.DictReader(open(md / "training_rows.csv")):
        comp[t["bucket"]][t["origin"]] += 1
    origins = ["positive", "hard_negative", "sibling_neg", "sampled_neg"]

    # per bucket: quality, PR curve, features
    buckets = []
    pooled_p, pooled_y = [], []
    for b in sorted(models):
        m = models[b]
        info = meta["buckets"][b]
        coef = m.lr.coef_[0]
        top = np.argsort(-coef)[:12]
        neg = np.argsort(coef)[:6]
        pooled_p += list(m.oof_prob); pooled_y += list(m.oof_y)
        buckets.append(dict(id=b, name=info["name"], parent=m.parent, definition=info.get("definition", ""), status=info.get("status", ""),
                            pos=m.n_pos, neg=m.n_neg, ap=round(m.ap, 4), brier=round(m.brier, 4), platt=[round(m.a, 4), round(m.b, 4)],
                            comp={o: comp[b][o] for o in origins},
                            pr=_pr(m.oof_y, m.oof_prob),
                            top=[[str(fz.names[i]), round(float(coef[i]), 4), _kind(str(fz.names[i]))] for i in top],
                            negf=[[str(fz.names[i]), round(float(coef[i]), 4), _kind(str(fz.names[i]))] for i in neg],
                            base_rate=round(m.n_pos / (m.n_pos + m.n_neg), 4)))
    pp, py = np.array(pooled_p), np.array(pooled_y)
    rel = []
    for lo in np.linspace(0, 1, 10, endpoint=False):
        sel = (pp >= lo) & (pp < lo + 0.1)
        if sel.any():
            rel.append(dict(lo=round(float(lo), 1), n=int(sel.sum()), pred=round(float(pp[sel].mean()), 4), obs=round(float(py[sel].mean()), 4)))
    ece = float(sum(r["n"] * abs(r["pred"] - r["obs"]) for r in rel) / max(1, len(pp)))
    nonconf = sorted(round(1 - float(p), 4) for p, y in zip(pp, py) if y == 1)

    # predictions
    pred = []
    for x in rows:
        r = rep[x["report_key"]]
        meta_r = r.get("meta") or {}
        pred.append(dict(key=x["report_key"], codes=x["codes"], decision=x["decision"], primary=x["primary"], p=x["primary_prob"],
                         set=x["prediction_set"], scores=x["scores"][:6], reason=x["reason"], evidence=x["evidence"],
                         feats=x["top_features"], date=meta_r.get("date", "") if isinstance(meta_r, dict) else "",
                         make=(meta_r.get("make", "") + " " + meta_r.get("model", "")).strip() if isinstance(meta_r, dict) else "",
                         text=text_of(r)[:1600], struct=structured_tokens(r)[:14]))
    faults, psum = [], None
    if predict_dir and (pathlib.Path(predict_dir) / "summary.json").exists():
        psum = json.loads((pathlib.Path(predict_dir) / "summary.json").read_text())
    if predict_dir and (pathlib.Path(predict_dir) / "fault_tests.csv").exists():
        faults = list(csv.DictReader(open(pathlib.Path(predict_dir) / "fault_tests.csv")))
    evaluation = None
    if eval_dir and (pathlib.Path(eval_dir) / "metrics.json").exists():
        ev = pathlib.Path(eval_dir)
        gold = gold_labels(gold_path) if gold_path else {}
        er = list(csv.DictReader(open(ev / "classifications.csv")))
        evaluation = dict(metrics=json.loads((ev / "metrics.json").read_text()),
                          per_bucket=list(csv.DictReader(open(ev / "per_bucket.csv"))) if (ev / "per_bucket.csv").exists() else [],
                          rows=[dict(key=e["report_key"], codes=e["cictt_codes"], decision=e["decision"], primary=e["primary"], p=e["primary_prob"],
                                     gold=sorted(gold.get(e["report_key"], [])), set=e["prediction_set"]) for e in er])
    dec = collections.Counter(x["decision"] for x in pred)
    by_parent = collections.defaultdict(collections.Counter)
    for x in pred:
        for c in (x["codes"].split() or ["none"])[:1]:
            by_parent[c][x["decision"]] += 1
    struct_vocab = sorted(fz.struct.vocabulary_, key=lambda t: fz.struct.idf_[fz.struct.vocabulary_[t]])[:24]
    data = dict(
        meta=dict(model_id=meta["model_id"], built=meta["built"], version=meta["taxclassify_version"], code_hash=meta["code_hash"],
                  run_id=meta["run_id"], run_hash=meta["run_output_hash"], taxonomy=meta["taxonomy"], settings=meta["settings"],
                  conformal=meta["conformal"], sources=meta["sources"], excluded=meta["excluded_reports"], skipped=meta["skipped"],
                  reports_file=str(reports_path)),
        features=dict(text_terms=len(fz.text.vocabulary_), struct_tokens=len(fz.struct.vocabulary_), struct_weight=fz.struct_weight,
                      struct_common=struct_vocab, total=len(fz.names)),
        training=dict(rows=sum(sum(c.values()) for c in comp.values()), by_origin={o: sum(c[o] for c in comp.values()) for o in origins}),
        buckets=buckets, reliability=dict(bins=rel, ece=round(ece, 4), n=len(pp)), nonconf=nonconf,
        predictions=pred, decisions=dict(dec), by_parent={k: dict(v) for k, v in sorted(by_parent.items())},
        set_sizes=dict(collections.Counter(len(x["set"]) for x in pred)), faults=faults, predict_summary=psum, checks=[dict(id=c, name=n) for c, n in CHECKS],
        evaluation=evaluation)
    return data


def write(data, out):
    tpl = (HERE / "demo_template.html").read_text()
    js = json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")
    pathlib.Path(out).write_text(tpl.replace("/*__DATA__*/null", js))
    return out
