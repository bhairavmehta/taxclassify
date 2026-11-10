"""Reading a taxground run or release, and building labelled training rows with lineage."""
import csv
import hashlib
import json
import pathlib
import random
import re


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_reports(path):
    """Reports in taxground's format: a JSON object with "reports" (or a bare list) of dicts holding at least
    report_key, l1_codes and narrative; findings, phase, meta and event_id are used when present."""
    doc = json.loads(pathlib.Path(path).read_text())
    reps = doc["reports"] if isinstance(doc, dict) else doc
    out = []
    for r in reps:
        codes = r.get("l1_codes") or []
        if isinstance(codes, str):                      # tolerate a stringified list
            codes = re.findall(r"[A-Z][A-Z\-]+", codes)
        out.append(dict(r, l1_codes=list(codes)))
    return out


class Run:
    """A taxground run folder: its results, its recorded inputs and its taxonomy."""

    def __init__(self, folder):
        self.dir = pathlib.Path(folder).resolve()
        self.results_path = self.dir / "grounding_results.json"
        G = json.loads(self.results_path.read_text())
        self.summary, self.settings, self.results = G["summary"], G["settings"], G["results"]
        self.run_id = self.summary["run_id"]
        self.reports_path = self.dir / "inputs" / "sample_reports.json"
        self.reports = {r["report_key"]: r for r in load_reports(self.reports_path)}
        self.taxonomy_path = self.dir / "taxonomy.json"

    def buckets(self, taxonomy=None, statuses=("grounded",)):
        """Buckets to classify into: those of a release taxonomy.json when given, else the run's buckets whose
        status is in `statuses`. Returns {bucket_id: dict(parent, name, definition, status)}."""
        out = {}
        if taxonomy:
            tax = json.loads(pathlib.Path(taxonomy).read_text())
            for p in tax["parents"]:
                for b in p["buckets"]:
                    out[b["bucket_id"]] = dict(parent=p["code"], name=b["name"], definition=b.get("definition", ""), status=b.get("status", ""))
            return out
        for r in self.results:
            par = (r.get("output") or {}).get("parent_code")
            if r["status"] in statuses and par:
                out[r["candidate_id"]] = dict(parent=par, name=r["name"], definition=r["definition"], status=r["status"])
        return out


def silver_labels(run, buckets, neg_per_pos=8, seed=7, exclude=()):
    """Training rows from the grounding run, each with its origin (lineage):
      positive      the bucket's verified support links (quote checked by G4)
      hard_negative a report the sibling rule gave to another bucket under the same parent
      sibling_neg   another bucket's support under the same parent
      sampled_neg   random reports under the same parent that no bucket of that parent cites
    Reports in `exclude` (hold-out or gold) are never used. Unlabelled reports are not true negatives (PU learning);
    sampling a few per positive and calibrating afterwards keeps that bias small, and gold labels measure it."""
    rnd = random.Random(seed)
    excl = set(exclude)
    res = {r["candidate_id"]: (i, r) for i, r in enumerate(run.results)}
    by_parent, pos = {}, {}
    for b, info in buckets.items():
        by_parent.setdefault(info["parent"], []).append(b)
        i, r = res.get(b, (None, None))
        sup = [s["report_key"] for s in ((r or {}).get("diag") or {}).get("support", [])] if r else []
        pos[b] = [k for k in sup if k in run.reports and k not in excl]
    cited = {p: {k for b in bs for k in pos[b]} for p, bs in by_parent.items()}
    under = {}
    for k, r in run.reports.items():
        if k in excl:
            continue
        for c in r["l1_codes"]:
            under.setdefault(c, []).append(k)
    rows = []
    for b, info in buckets.items():
        p = info["parent"]
        i, r = res.get(b, (None, None))
        loc = f"{run.results_path.name}#results[{i}]" if i is not None else "taxonomy"
        for j, k in enumerate(pos[b]):
            rows.append(dict(bucket=b, report_key=k, y=1, weight=1.0, origin="positive", source=f"{loc}.diag.support[{j}]"))
        seen = set(pos[b])
        for j, m in enumerate(((r or {}).get("diag") or {}).get("moved_to_sibling", [])):
            k = m["report_key"]
            if k in run.reports and k not in excl and k not in seen:
                rows.append(dict(bucket=b, report_key=k, y=0, weight=1.0, origin="hard_negative", source=f"{loc}.diag.moved_to_sibling[{j}]"))
                seen.add(k)
        for sib in by_parent[p]:
            if sib == b:
                continue
            for k in pos[sib]:
                if k not in seen:
                    rows.append(dict(bucket=b, report_key=k, y=0, weight=1.0, origin="sibling_neg", source=f"support of {sib}"))
                    seen.add(k)
        pool = sorted(k for k in under.get(p, []) if k not in cited.get(p, set()) and k not in seen)
        rnd.shuffle(pool)
        for k in pool[: max(10, neg_per_pos * len(pos[b]))]:
            rows.append(dict(bucket=b, report_key=k, y=0, weight=1.0, origin="sampled_neg", source=f"reports coded {p}, not cited"))
    return rows


def gold_labels(path):
    """Gold labels: a CSV with report_key and bucket (one row per label; bucket empty or NONE = none of these), or a
    taxground test-report JSON whose reports carry _expected_bucket. Returns {report_key: set(buckets)}."""
    p = pathlib.Path(path)
    gold = {}
    if p.suffix == ".json":
        for r in load_reports(p):
            if "_expected_bucket" in r:
                v = r["_expected_bucket"]
                gold.setdefault(r["report_key"], set()).update([v] if v and v != "NONE" else [])
        return gold
    for row in csv.DictReader(open(p, encoding="utf-8")):
        b = (row.get("bucket") or "").strip()
        gold.setdefault(row["report_key"].strip(), set()).update([b] if b and b.upper() != "NONE" else [])
    return gold
