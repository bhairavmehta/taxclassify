"""Deterministic checks on every classification, the classifier's counterpart of G1–G10, and fault tests that
prove each check blocks what it should."""
import copy
import math

CHECKS = [("C1", "Assigned bucket exists in the version"),
          ("C2", "Bucket sits under one of the report's CICTT codes"),
          ("C3", "Evidence sentence appears word for word in the report"),
          ("C4", "Probabilities are numbers in [0, 1]"),
          ("C5", "Every non-assignment carries a reason")]


def check(row, buckets, report_text, report_codes):
    out = []
    assigned = [b for b in row["labels"] if b]
    out.append(("C1", all(b in buckets for b in assigned) and (row["primary"] in buckets or not row["primary"])))
    out.append(("C2", all(buckets[b]["parent"] in report_codes for b in assigned if b in buckets)))
    out.append(("C3", (not assigned) or (bool(row["evidence"]) and row["evidence"] in report_text)))
    probs = [p for _, p in row["scores"]]
    out.append(("C4", all(isinstance(p, float) and not math.isnan(p) and 0.0 <= p <= 1.0 for p in probs)))
    out.append(("C5", row["decision"] == "assign" or bool(row["reason"])))
    return out


def fault_tests(row, buckets, text, codes):
    """Inject the faults an unconstrained model might produce into a valid row; each must fail its check."""
    faults = []
    f = copy.deepcopy(row); f["labels"] = ["X99"]; f["primary"] = "X99"; faults.append(("Invented bucket", "C1", f))
    f = copy.deepcopy(row); f["evidence"] = "The aircraft was struck by a meteorite on short final."; faults.append(("Misquoted evidence", "C3", f))
    wrong = next((b for b, i in buckets.items() if i["parent"] not in codes), None)
    if wrong:
        f = copy.deepcopy(row); f["labels"] = [wrong]; f["primary"] = wrong; faults.append(("Bucket under another parent", "C2", f))
    f = copy.deepcopy(row); f["scores"] = [(b, 1.7) for b, _ in row["scores"]] or [("X", 1.7)]; faults.append(("Probability above 1", "C4", f))
    f = copy.deepcopy(row); f["decision"] = "abstain"; f["reason"] = ""; faults.append(("Abstain without reason", "C5", f))
    out = []
    for name, cid, bad in faults:
        res = dict(check(bad, buckets, text, codes))
        out.append(dict(test=name, check=cid, blocked=not res[cid]))
    return out
