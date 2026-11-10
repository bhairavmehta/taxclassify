"""Stage-1 classifier: one calibrated logistic regression per bucket, trained against its siblings under the same
CICTT parent, plus a split-conformal threshold that turns probabilities into prediction sets."""
import math
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

DEFAULTS = dict(C=1.0, folds=3, alpha=0.10, min_positives=4, min_prob=0.5, neg_per_pos=8, seed=7,
                max_features=200000, struct_weight=0.5)


def _platt(dec, y, seed):
    """Platt scaling: a one-feature logistic regression from decision value to probability."""
    lr = LogisticRegression(C=1e6, solver="lbfgs", random_state=seed)
    lr.fit(dec.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _prob(dec, a, b):
    return 1.0 / (1.0 + np.exp(-(a * dec + b)))


class BucketModel:
    def __init__(self, bucket, parent):
        self.bucket, self.parent = bucket, parent

    def fit(self, X, y, w, S):
        self.n_pos, self.n_neg = int(y.sum()), int(len(y) - y.sum())
        k = max(2, min(S["folds"], self.n_pos, self.n_neg))   # every fold must hold both classes
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=S["seed"]).split(X, y):
            m = LogisticRegression(C=S["C"], class_weight="balanced", solver="liblinear", random_state=S["seed"], max_iter=2000)
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            oof[te] = m.decision_function(X[te])
        self.a, self.b = _platt(oof, y, S["seed"])
        self.oof_prob = _prob(oof, self.a, self.b)
        self.oof_y = y
        self.lr = LogisticRegression(C=S["C"], class_weight="balanced", solver="liblinear", random_state=S["seed"], max_iter=2000)
        self.lr.fit(X, y, sample_weight=w)
        self.ap = float(average_precision_score(y, self.oof_prob)) if 0 < y.sum() < len(y) else float("nan")
        self.brier = float(brier_score_loss(y, self.oof_prob))
        return self

    def prob(self, X):
        return _prob(self.lr.decision_function(X), self.a, self.b)

    def contributions(self, x, names, k=5):
        """Top features pushing this report toward the bucket: coefficient × feature value."""
        row = x.tocsr()
        idx, val = row.indices, row.data
        if not len(idx):
            return []
        c = self.lr.coef_[0, idx] * val
        order = np.argsort(-c)[:k]
        return [(str(names[idx[i]]), round(float(c[i]), 4)) for i in order if c[i] > 0]


def conformal_threshold(models, alpha):
    """Split-conformal on out-of-fold probabilities of each positive's own bucket: nonconformity = 1 − p(true).
    A bucket enters a report's prediction set when p ≥ 1 − q̂; on exchangeable data the set holds the true bucket
    with probability ≥ 1 − alpha. Computed on silver labels until gold labels exist, so it is a target, not a guarantee."""
    scores = sorted(1.0 - float(p) for m in models.values() for p, y in zip(m.oof_prob, m.oof_y) if y == 1)
    n = len(scores)
    if not n:
        return 0.5, 0
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(scores[k - 1]), n


SENT = re.compile(r"[^.!?]+(?:[.!?]+[\"”’)]*|$)")


def evidence(model, featurizer, text):
    """The sentence of the report that most supports the bucket under the bucket's own linear model; returned
    verbatim (a substring of the report text), so check C3 can verify it word for word."""
    sents = [s.strip() for s in SENT.findall(text or "") if len(s.strip()) > 20]
    if not sents:
        return ""
    dec = model.lr.decision_function(featurizer.transform_text(sents))
    return sents[int(np.argmax(dec))]
