"""Features: report text (TF-IDF words and word pairs) plus NTSB structure (finding-code prefixes, phase, aircraft
category, CICTT codes). Deterministic: the same corpus gives the same vocabulary and the same matrix."""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

FIND = re.compile(r"(Cause|Contributing factor|Finding): ([^.]+)")


def text_of(r):
    return (r.get("narrative") or "") + "\n" + (r.get("findings") or "")


def structured_tokens(r):
    """Coded fields as tokens: the first three levels of every NTSB finding with its role, the flight phase, the
    aircraft category and the CICTT codes. These carry signal the narrative may not spell out."""
    toks = []
    for role, path in FIND.findall(r.get("findings") or ""):
        parts = [p.strip() for p in path.split("-") if p.strip()]
        for depth in (2, 3):
            if len(parts) >= depth:
                toks.append(f"F:{role[:5]}:{'/'.join(parts[:depth])}")
    meta = r.get("meta") or {}
    if isinstance(meta, dict):
        if meta.get("category"):
            toks.append(f"CAT:{meta['category']}")
    if r.get("phase"):
        toks.append(f"PH:{str(r['phase']).split('-')[0]}")
    toks += [f"L1:{c}" for c in r.get("l1_codes", [])]
    return toks


def _tokens(toks):
    """The structured fields are already tokens (a module-level function, so the model can be saved)."""
    return toks


class Featurizer:
    def __init__(self, max_features=200000, min_df=2, struct_weight=0.5):
        self.text = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=min_df,
                                    max_features=max_features, dtype=np.float32)
        self.struct = TfidfVectorizer(analyzer=_tokens, min_df=2, dtype=np.float32)
        self.struct_weight = struct_weight

    def fit(self, reports):
        self.text.fit([text_of(r) for r in reports])
        self.struct.fit([structured_tokens(r) for r in reports])
        self.names = np.array(list(self.text.get_feature_names_out()) + list(self.struct.get_feature_names_out()))
        return self

    def transform(self, reports):
        a = self.text.transform([text_of(r) for r in reports])
        b = self.struct.transform([structured_tokens(r) for r in reports]) * self.struct_weight
        return sparse.hstack([a, b]).tocsr()

    def transform_text(self, texts):
        """Sentences scored in the same space (text part only, structure zero), for evidence and explanation."""
        a = self.text.transform(texts)
        return sparse.hstack([a, sparse.csr_matrix((a.shape[0], len(self.struct.vocabulary_)), dtype=np.float32)]).tocsr()
