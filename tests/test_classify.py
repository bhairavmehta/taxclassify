"""taxclassify unit tests: no database. Uses the synthetic combined run when present (out/run-combined-*)."""
import glob
import pathlib
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from taxclassify import cli                                   # noqa: E402
from taxclassify.checks import check, fault_tests             # noqa: E402
from taxclassify.features import structured_tokens            # noqa: E402
from taxclassify.model import conformal_threshold             # noqa: E402

RUNS = sorted(glob.glob(str(ROOT / "out" / "run-combined-*"))) or sorted(glob.glob(str(ROOT / "classifier_input" / "synthetic_run")))
TEST = ROOT / "data/test_reports.json" if (ROOT / "data/test_reports.json").exists() else ROOT / "classifier_input/synthetic_test_reports.json"


def test_structured_tokens_read_ntsb_findings():
    r = dict(findings="Cause: Aircraft-Aircraft systems-Ice/rain protection system-Intake anti-ice. Finding: Personnel issues-Action/decision-Action-Lack of action.",
             phase="Landing-flare", l1_codes=["SCF-PP"], meta={"category": "AIR"})
    t = structured_tokens(r)
    assert "F:Cause:Aircraft/Aircraft systems" in t and "F:Cause:Aircraft/Aircraft systems/Ice/rain protection system" in t
    assert "PH:Landing" in t and "L1:SCF-PP" in t and "CAT:AIR" in t


def test_checks_block_every_fault():
    buckets = {"B1": dict(parent="RI"), "B2": dict(parent="RE")}
    row = dict(labels=["B1"], primary="B1", scores=[("B1", 0.9)], decision="assign", reason="", evidence="The aircraft crossed the hold line.")
    text = "Probable cause: x. The aircraft crossed the hold line. More."
    assert all(ok for _, ok in check(row, buckets, text, ["RI"]))
    res = fault_tests(row, buckets, text, ["RI"])
    assert len(res) == 5 and all(f["blocked"] for f in res)


def test_conformal_threshold_is_a_quantile():
    class M:
        def __init__(self, p, y):
            self.oof_prob, self.oof_y = p, y
    q, n = conformal_threshold({"a": M([0.9, 0.8, 0.7, 0.2], [1, 1, 1, 0]), "b": M([0.6], [1])}, 0.25)
    assert n == 4 and abs(q - 0.4) < 1e-9          # scores 0.1 0.2 0.3 0.4 → k = ceil(5 × 0.75) = 4 → 0.4


@pytest.mark.skipif(not RUNS, reason="no synthetic combined run (run demo.sh or build step 0 first)")
def test_train_predict_evaluate_on_synthetic():
    with tempfile.TemporaryDirectory() as t:
        cli.main(["train", "--run", RUNS[-1], "--include-review", "--out", t, "--set", "min_positives=3"])
        model = sorted(pathlib.Path(t).glob("clf-*"))[0]
        assert (model / "model_card.md").exists() and (model / "training_rows.csv").exists()
        cli.main(["evaluate", "--model", str(model), "--reports", str(TEST),
                  "--gold", str(TEST), "--out", t + "/eval"])
        import json
        m = json.loads(pathlib.Path(t, "eval", "metrics.json").read_text())
        assert m["assigned_precision"] is None or m["assigned_precision"] >= 0.8
        s = json.loads(pathlib.Path(t, "eval", "summary.json").read_text())
        assert s["check_failures"] == 0 and s["faults_blocked"].split("/")[0] == s["faults_blocked"].split("/")[1]
