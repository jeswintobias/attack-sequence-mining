"""Adapters presenting existing models through one `fit`/`rank` interface.

Each adapter must be constructible fresh per fold and must fit using only the
rows it is handed, so nothing crosses a fold boundary. The singleton unigram
prior is the one exception: it is estimated from length-1 rows of the raw CSV,
which are disjoint from the length>=4 sequences used for modelling, so it
carries no fold information and is cached to avoid re-reading a 1M-row file
once per fold.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"), str(SRC / "mining")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from baselines import MajorityBaseline, MarkovOrder1, MarkovOrder2, build_singleton_prior
from prefixspan_predictor import PrefixSpanPredictor


@lru_cache(maxsize=1)
def _singleton_prior():
    """Unigram prior over length-1 sightings; disjoint from the modelled sequences."""
    return build_singleton_prior(config.SIGHTINGS_DATA_PATH)


def _sequences_with_weights(examples: pd.DataFrame):
    """Recover unique full sequences and their duplicate counts from prefix rows."""
    frame = pd.DataFrame({
        "full_seq": examples["full_seq"].map(tuple),
        "dup_count": examples["dup_count"].to_numpy(),
    }).drop_duplicates("full_seq")
    return frame["full_seq"].tolist(), frame["dup_count"].astype(int).tolist()


class MajorityAdapter:
    name = "B1_Majority"

    def fit(self, examples):
        self.model = MajorityBaseline()
        self.model.fit(examples)
        return self

    def rank(self, examples):
        ranked = self.model.predict(None)
        return [ranked] * len(examples)


class Markov2Adapter:
    """Second-order Markov with back-off; the strongest non-learned baseline."""

    name = "B4_Markov2"

    def __init__(self, m: int = 5, k: float = 0.001):
        self.m, self.k = m, k

    def fit(self, examples):
        prior = _singleton_prior()
        self.m1 = MarkovOrder1(k=self.k, fallback_counts=prior)
        self.m1.fit(examples)
        self.m2 = MarkovOrder2(m=self.m, k1=self.k, fallback_counts=prior)
        self.m2.fit(examples, markov1=self.m1)
        return self

    def rank(self, examples):
        return [self.m2.predict(p) for p in examples["prefix"]]


class PrefixSpanAdapter:
    """Mined sequential rules with Markov-2 back-off."""

    name = "B5_PrefixSpan"

    def __init__(self, min_support_pct: float = 0.01, min_conf: float = 0.7, max_len: int = 4):
        self.min_support_pct, self.min_conf, self.max_len = min_support_pct, min_conf, max_len

    def fit(self, examples):
        self.backbone = Markov2Adapter().fit(examples)
        sequences, weights = _sequences_with_weights(examples)
        self.model = PrefixSpanPredictor(
            min_support_pct=self.min_support_pct,
            min_conf=self.min_conf,
            max_len=self.max_len,
            markov2_fallback=self.backbone.m2,
            longest_first=True,
        )
        self.model.fit(sequences, weights)
        return self

    def rank(self, examples):
        return [self.model.predict(p) for p in examples["prefix"]]


class SklearnProbaAdapter:
    """Any scikit-learn-style classifier over the shared prefix feature block.

    The technique vocabulary and the label encoding are rebuilt from the fold's
    own fit rows, so a fold never sees a column or a class defined by held-out
    data.
    """

    def __init__(self, estimator_factory, name: str, n_estimators_attr: str | None = None):
        self.estimator_factory = estimator_factory
        self.name = name
        self.n_estimators_attr = n_estimators_attr

    def fit(self, examples):
        from sklearn.preprocessing import LabelEncoder
        from lightgbm_predictor import _make_features, _sequence_weights

        self.encoder = LabelEncoder().fit(examples["next"].astype(str))
        y = self.encoder.transform(examples["next"].astype(str))
        self.techniques = sorted(
            set(examples["next"].astype(str)) | {t for p in examples["prefix"] for t in p}
        )
        X = _make_features(examples, self.techniques)
        self.estimator = self.estimator_factory()
        self.estimator.fit(X, y, sample_weight=_sequence_weights(examples))
        return self

    def rank(self, examples):
        from lightgbm_predictor import _make_features

        X = _make_features(examples, self.techniques)
        probabilities = self.estimator.predict_proba(X)
        classes = np.asarray(self.estimator.classes_, dtype=int)
        names = np.asarray(self.encoder.classes_, dtype=object)[classes]
        order = np.argsort(-probabilities, axis=1, kind="stable")
        return [names[row].tolist() for row in order]
