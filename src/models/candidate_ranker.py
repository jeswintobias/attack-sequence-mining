"""Learning-to-rank over (prefix, candidate) pairs with mining-derived features.

Why this formulation
--------------------
The existing boosted-tree models solve a 94-class softprob problem over 511
sparse binary features, with 8,329 training rows and 47 classes holding fewer
than 5 examples each. Every boosting round fits 94 one-vs-rest functions that
the data cannot support, and a pattern indicator such as "rule P matched" is a
property of the *prefix*, so each of those 94 heads has to rediscover what P
implies on its own.

Here each row is one (prefix, candidate) pair and the objective is a single
ranking function. Pattern and transition evidence becomes *candidate-specific*:
the Markov probability of this candidate, the confidence of the mined rules that
predict it, its co-occurrence lift with the prefix. Alongside those we pass the
counts and entropies the evidence rests on, so the model can learn *when* to
trust each source rather than trusting it uniformly. That last point is the
measured motivation: Markov-2 is uniquely correct on 8.1% of the examples
LightGBM misses, yet blending the two with a single global weight recovers only
+0.0036 MRR, so the combination has to be conditional to be worth anything.

Mining is refitted inside every fold from that fold's rows only.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from functools import lru_cache
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"), str(SRC / "mining")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from baselines import build_singleton_prior
from prefixspan import prefixspan_mine

FEATURES = [
    "m1_prob", "m1_count", "m1_ctx_total", "m1_entropy",
    "m2_prob", "m2_count", "m2_ctx_total", "m2_entropy", "m2_has_context",
    "m3_prob", "m3_count", "m3_has_context",
    "ps_max_conf", "ps_max_support", "ps_best_ante_len", "ps_n_rules",
    "lift_max", "lift_mean", "cooc_max", "cooc_frac",
    # Prefix-content evidence. Without these the model sees only the last one or
    # two techniques and collapses into a reweighted Markov-2, which is what the
    # first version did (m2_prob gain 400 against 10.8 for everything else).
    # These aggregate transition evidence over every prefix member, which is the
    # information the 511 seen_*/last-k indicators give the softprob models.
    "bag_trans_mean", "bag_trans_max", "bag_trans_sum",
    "lift_last1", "lift_last2",
    "cand_prior", "cand_label_freq", "prefix_len",
]

ADD_K = 0.001


@lru_cache(maxsize=1)
def _singleton_prior():
    return build_singleton_prior(config.SIGHTINGS_DATA_PATH)


def _entropy(counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * log(c / total) for c in counter.values() if c > 0)


def _is_subsequence(sub, seq) -> bool:
    it = iter(seq)
    return all(x in it for x in sub)


class CandidateFeatureBuilder:
    """Fits transition tables and pattern rules, then scores candidate pairs."""

    def __init__(self, min_support_pct: float = 0.01, min_conf: float = 0.5, max_len: int = 4):
        self.min_support_pct = min_support_pct
        self.min_conf = min_conf
        self.max_len = max_len

    # ---- fitting -----------------------------------------------------------
    def fit(self, examples: pd.DataFrame):
        weights = examples["dup_count"].to_numpy(dtype=float)

        self.trans1 = defaultdict(Counter)
        self.trans2 = defaultdict(Counter)
        self.trans3 = defaultdict(Counter)
        self.label_freq = Counter()
        for prefix, nxt, w in zip(examples["prefix"], examples["next"], weights):
            self.trans1[prefix[-1]][nxt] += w
            if len(prefix) >= 2:
                self.trans2[(prefix[-2], prefix[-1])][nxt] += w
            if len(prefix) >= 3:
                self.trans3[(prefix[-3], prefix[-2], prefix[-1])][nxt] += w
            self.label_freq[nxt] += w

        self.candidates = sorted(self.label_freq)
        self.cand_index = {c: i for i, c in enumerate(self.candidates)}
        self.m1_entropy = {k: _entropy(v) for k, v in self.trans1.items()}
        self.m2_entropy = {k: _entropy(v) for k, v in self.trans2.items()}
        self.m1_total = {k: sum(v.values()) for k, v in self.trans1.items()}
        self.m2_total = {k: sum(v.values()) for k, v in self.trans2.items()}
        self.m3_total = {k: sum(v.values()) for k, v in self.trans3.items()}
        # Row-normalised first-order transitions, reused by the bag-of-prefix
        # features so every prefix member contributes its own transition evidence.
        self.trans1_norm = {
            context: {t: c / self.m1_total[context] for t, c in counts.items()}
            for context, counts in self.trans1.items()
        }

        self._fit_patterns(examples)
        self._fit_cooccurrence(examples)

        prior = _singleton_prior()
        prior_total = sum(prior.values()) or 1
        self.cand_prior = np.array(
            [prior.get(c, 0) / prior_total for c in self.candidates], dtype=np.float32
        )
        freq_total = sum(self.label_freq.values()) or 1
        self.cand_freq = np.array(
            [self.label_freq[c] / freq_total for c in self.candidates], dtype=np.float32
        )
        return self

    def _fit_patterns(self, examples: pd.DataFrame):
        """Mine sequential rules from this fold's full sequences."""
        unique = pd.DataFrame({
            "full_seq": examples["full_seq"].map(tuple),
            "dup_count": examples["dup_count"].to_numpy(),
        }).drop_duplicates("full_seq")
        sequences = unique["full_seq"].tolist()
        weights = unique["dup_count"].astype(int).tolist()
        total_weight = sum(weights)
        patterns = prefixspan_mine(
            sequences,
            min_support=int(total_weight * self.min_support_pct),
            max_len=self.max_len,
            weights=weights,
        )
        support = {tuple(p): s for s, p in patterns}
        # Index rules by the final antecedent item: a rule can only match a
        # prefix whose last technique equals that item, so this avoids scanning
        # every rule for every row.
        self.rules_by_anchor = defaultdict(list)
        for pattern, s in support.items():
            if len(pattern) < 2:
                continue
            antecedent, consequent = tuple(pattern[:-1]), pattern[-1]
            conf = s / support.get(antecedent, total_weight)
            if conf >= self.min_conf:
                self.rules_by_anchor[antecedent[-1]].append(
                    (antecedent, consequent, conf, s)
                )

    def _fit_cooccurrence(self, examples: pd.DataFrame):
        """Pairwise lift over this fold's unique sequences (FP-Growth statistics)."""
        unique = pd.DataFrame({
            "full_seq": examples["full_seq"].map(tuple),
            "dup_count": examples["dup_count"].to_numpy(),
        }).drop_duplicates("full_seq")
        item_count = Counter()
        pair_count = defaultdict(Counter)
        n_tx = 0
        for seq, w in zip(unique["full_seq"], unique["dup_count"]):
            w = float(w)
            n_tx += w
            items = sorted(set(seq))
            for a in items:
                item_count[a] += w
            for i, a in enumerate(items):
                for b in items[i + 1:]:
                    pair_count[a][b] += w
                    pair_count[b][a] += w
        self.n_transactions = n_tx or 1.0
        self.item_count = item_count
        self.pair_count = pair_count

    # ---- transforming ------------------------------------------------------
    def transform(self, examples: pd.DataFrame):
        """Build the (rows x candidates) feature block.

        Returns (X, qid, n_candidates). Row order is prefix-major, so `qid` is
        already sorted ascending as XGBRanker requires.
        """
        n_rows, n_cand = len(examples), len(self.candidates)
        grid = {name: np.zeros((n_rows, n_cand), dtype=np.float32) for name in FEATURES}
        # Candidate-level constants broadcast across every row.
        grid["cand_prior"][:] = self.cand_prior
        grid["cand_label_freq"][:] = self.cand_freq

        for row, prefix in enumerate(examples["prefix"]):
            prefix = tuple(prefix)
            grid["prefix_len"][row, :] = len(prefix)
            self._fill_markov(grid, row, prefix, n_cand)
            self._fill_patterns(grid, row, prefix)
            self._fill_cooccurrence(grid, row, prefix)
            self._fill_prefix_content(grid, row, prefix)

        X = np.stack([grid[name] for name in FEATURES], axis=-1).reshape(-1, len(FEATURES))
        qid = np.repeat(np.arange(n_rows, dtype=np.int32), n_cand)
        return X, qid, n_cand

    def _fill_markov(self, grid, row, prefix, n_cand):
        last = prefix[-1]
        counts1 = self.trans1.get(last)
        if counts1 is not None:
            total = self.m1_total[last]
            grid["m1_ctx_total"][row, :] = total
            grid["m1_entropy"][row, :] = self.m1_entropy[last]
            denom = total + ADD_K * n_cand
            grid["m1_prob"][row, :] = ADD_K / denom
            for technique, count in counts1.items():
                j = self.cand_index.get(technique)
                if j is not None:
                    grid["m1_count"][row, j] = count
                    grid["m1_prob"][row, j] = (count + ADD_K) / denom
        if len(prefix) >= 2:
            context = (prefix[-2], prefix[-1])
            counts2 = self.trans2.get(context)
            if counts2 is not None:
                total = self.m2_total[context]
                grid["m2_has_context"][row, :] = 1.0
                grid["m2_ctx_total"][row, :] = total
                grid["m2_entropy"][row, :] = self.m2_entropy[context]
                denom = total + ADD_K * n_cand
                grid["m2_prob"][row, :] = ADD_K / denom
                for technique, count in counts2.items():
                    j = self.cand_index.get(technique)
                    if j is not None:
                        grid["m2_count"][row, j] = count
                        grid["m2_prob"][row, j] = (count + ADD_K) / denom
        if len(prefix) >= 3:
            context = (prefix[-3], prefix[-2], prefix[-1])
            counts3 = self.trans3.get(context)
            if counts3 is not None:
                total = self.m3_total[context]
                grid["m3_has_context"][row, :] = 1.0
                denom = total + ADD_K * n_cand
                grid["m3_prob"][row, :] = ADD_K / denom
                for technique, count in counts3.items():
                    j = self.cand_index.get(technique)
                    if j is not None:
                        grid["m3_count"][row, j] = count
                        grid["m3_prob"][row, j] = (count + ADD_K) / denom

    def _fill_prefix_content(self, grid, row, prefix):
        """Aggregate first-order transition evidence over every prefix member."""
        n_cand = len(self.candidates)
        totals = np.zeros(n_cand, dtype=np.float32)
        maxima = np.zeros(n_cand, dtype=np.float32)
        members = set(prefix)
        for item in members:
            probabilities = self.trans1_norm.get(item)
            if not probabilities:
                continue
            for technique, probability in probabilities.items():
                j = self.cand_index.get(technique)
                if j is None:
                    continue
                totals[j] += probability
                if probability > maxima[j]:
                    maxima[j] = probability
        grid["bag_trans_sum"][row, :] = totals
        grid["bag_trans_max"][row, :] = maxima
        grid["bag_trans_mean"][row, :] = totals / max(len(members), 1)

    def _fill_patterns(self, grid, row, prefix):
        head = prefix[:-1]
        for antecedent, consequent, conf, support in self.rules_by_anchor.get(prefix[-1], ()):
            if len(antecedent) > 1 and not _is_subsequence(antecedent[:-1], head):
                continue
            j = self.cand_index.get(consequent)
            if j is None:
                continue
            grid["ps_n_rules"][row, j] += 1.0
            if conf > grid["ps_max_conf"][row, j]:
                grid["ps_max_conf"][row, j] = conf
            if support > grid["ps_max_support"][row, j]:
                grid["ps_max_support"][row, j] = support
            if len(antecedent) > grid["ps_best_ante_len"][row, j]:
                grid["ps_best_ante_len"][row, j] = len(antecedent)

    def _fill_cooccurrence(self, grid, row, prefix):
        """Max/mean lift and max co-occurrence count against the prefix items."""
        totals = np.zeros(len(self.candidates), dtype=np.float32)
        seen = np.zeros(len(self.candidates), dtype=np.float32)
        for item in set(prefix):
            partners = self.pair_count.get(item)
            if not partners:
                continue
            count_a = self.item_count.get(item, 0.0)
            if count_a <= 0:
                continue
            for partner, pair in partners.items():
                j = self.cand_index.get(partner)
                if j is None:
                    continue
                count_b = self.item_count.get(partner, 0.0)
                if count_b <= 0:
                    continue
                lift = pair * self.n_transactions / (count_a * count_b)
                totals[j] += lift
                seen[j] += 1.0
                if lift > grid["lift_max"][row, j]:
                    grid["lift_max"][row, j] = lift
                if pair > grid["cooc_max"][row, j]:
                    grid["cooc_max"][row, j] = pair
        nonzero = seen > 0
        grid["lift_mean"][row, nonzero] = totals[nonzero] / seen[nonzero]
        grid["cooc_frac"][row, :] = seen / max(len(set(prefix)), 1)
        # Lift against the two most recent techniques specifically, so recency is
        # distinguishable from bulk co-occurrence.
        for offset, feature in ((1, "lift_last1"), (2, "lift_last2")):
            if len(prefix) < offset:
                continue
            item = prefix[-offset]
            partners = self.pair_count.get(item)
            count_a = self.item_count.get(item, 0.0)
            if not partners or count_a <= 0:
                continue
            for partner, pair in partners.items():
                j = self.cand_index.get(partner)
                if j is None:
                    continue
                count_b = self.item_count.get(partner, 0.0)
                if count_b > 0:
                    grid[feature][row, j] = pair * self.n_transactions / (count_a * count_b)

    def labels(self, examples: pd.DataFrame) -> np.ndarray:
        """Binary relevance: 1 for the observed next technique, 0 otherwise."""
        n_cand = len(self.candidates)
        y = np.zeros((len(examples), n_cand), dtype=np.int32)
        for row, target in enumerate(examples["next"].astype(str)):
            j = self.cand_index.get(target)
            if j is not None:
                y[row, j] = 1
        return y.reshape(-1)


class CandidateRankerAdapter:
    """XGBRanker over (prefix, candidate) pairs; plugs into the CV harness."""

    def __init__(self, name: str = "CandidateRanker", n_estimators: int = 300,
                 learning_rate: float = 0.08, max_depth: int = 6, **params):
        self.name = name
        self.params = {
            "objective": "rank:ndcg",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "tree_method": "hist",
            "random_state": config.RANDOM_SEED,
            "n_jobs": 1,
            "verbosity": 0,
            **params,
        }

    def fit(self, examples: pd.DataFrame):
        import xgboost as xgb

        self.builder = CandidateFeatureBuilder().fit(examples)
        X, qid, _ = self.builder.transform(examples)
        y = self.builder.labels(examples)
        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X, y, qid=qid)
        return self

    def rank(self, examples: pd.DataFrame):
        X, _, n_cand = self.builder.transform(examples)
        scores = self.model.predict(X).reshape(len(examples), n_cand)
        names = np.asarray(self.builder.candidates, dtype=object)
        order = np.argsort(-scores, axis=1, kind="stable")
        return [names[row].tolist() for row in order]

    def feature_importance(self) -> pd.DataFrame:
        gains = self.model.get_booster().get_score(importance_type="gain")
        rows = [{"feature": FEATURES[int(k[1:])], "gain": v} for k, v in gains.items()]
        return pd.DataFrame(rows).sort_values("gain", ascending=False)
