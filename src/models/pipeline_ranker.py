"""Full pipeline: graph (A) + tactics (B) + interactions (C) + masking (D) -> XGBoost.

Run from the repository root:
  python src/models/pipeline_ranker.py                 # fold-0 ablation
  python src/models/pipeline_ranker.py --full          # 5 folds + paired CIs

Stages, in the order they are applied to a prefix:

  A  graph         Node2Vec embedding closeness, hop distance, and -log-probability
                   path cost from the prefix to the candidate.
  B  tactics       candidate tactic multi-hot, tactic-level transition probability,
                   overlap with the prefix's tactics, kill-chain progression.
  C  interactions  recency-decayed transition evidence over prefix positions and
                   subsequence-alignment breadth for the candidate.
  D  masking       candidates already in the prefix are removed before ranking.
                   Applied centrally by the CV harness, so it cannot be skipped.
  XGBoost          XGBRanker over the assembled (prefix, candidate) rows.

Groups are individually switchable so each one's contribution can be measured
rather than assumed. Every table, graph, embedding and rule set is refit inside
each fold from that fold's rows only. The test split is never loaded.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"),
          str(SRC / "mining"), str(SRC / "eval")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from candidate_ranker import FEATURES as BASE_FEATURES, CandidateFeatureBuilder
from graph_features import TransitionGraph
from tactic_map import TACTIC_ORDER, load_mapping, multi_hot, tactic_columns

EMBED_DIM = 16
DECAY_LAMBDAS = (0.5, 0.8)

GRAPH_FEATURES = (
    ["emb_cos_last", "emb_cos_mean", "emb_cos_max",
     "hops_from_last", "hops_min_prefix",
     "path_cost_last", "path_cost_min"]
    + [f"emb_cand_{i}" for i in range(EMBED_DIM)]
)
TACTIC_FEATURES = (
    [f"cand_{c}" for c in tactic_columns()]
    + ["tactic_trans_prob", "tactic_trans_count", "tactic_overlap_prefix",
       "tactic_is_new", "tactic_index_delta"]
)
INTERACTION_FEATURES = [
    f"recency_decay_{str(l).replace('.', '')}" for l in DECAY_LAMBDAS
] + ["nearest_contributor_pos", "n_prefix_transitions", "frac_prefix_transitions",
     "recency_weighted_lift"]


class PipelineFeatureBuilder:
    """Assembles the base mining block plus the A/B/C groups."""

    def __init__(self, use_graph: bool = True, use_tactics: bool = True,
                 use_interactions: bool = True, embed_dim: int = EMBED_DIM,
                 min_conf: float = 0.3):
        self.use_graph = use_graph
        self.use_tactics = use_tactics
        self.use_interactions = use_interactions
        self.embed_dim = embed_dim
        self.base = CandidateFeatureBuilder(min_conf=min_conf)

    @property
    def feature_names(self) -> list[str]:
        names = list(BASE_FEATURES)
        if self.use_graph:
            names += GRAPH_FEATURES
        if self.use_tactics:
            names += TACTIC_FEATURES
        if self.use_interactions:
            names += INTERACTION_FEATURES
        return names

    # ---- fit ---------------------------------------------------------------
    def fit(self, examples: pd.DataFrame):
        self.base.fit(examples)
        self.candidates = self.base.candidates
        self.cand_index = self.base.cand_index
        n_cand = len(self.candidates)

        if self.use_graph:
            self.graph = TransitionGraph(dim=self.embed_dim).fit(examples)
            self.cand_graph_row = self.graph.candidate_rows(self.candidates)
            self.cand_valid = self.cand_graph_row >= 0
            self.cand_unit = np.zeros((n_cand, self.embed_dim), dtype=np.float32)
            self.cand_embed = np.zeros((n_cand, self.embed_dim), dtype=np.float32)
            rows = self.cand_graph_row[self.cand_valid]
            self.cand_unit[self.cand_valid] = self.graph.unit[rows]
            self.cand_embed[self.cand_valid] = self.graph.embedding[rows]

        if self.use_tactics:
            self.mapping = load_mapping()
            self.tactic_index = {t: i for i, t in enumerate(TACTIC_ORDER)}
            self.cand_tactics = multi_hot(self.candidates, self.mapping)
            self._fit_tactic_transitions(examples, n_cand)
        return self

    def _fit_tactic_transitions(self, examples: pd.DataFrame, n_cand: int):
        """Tactic-level transition counts, shared across all techniques in a tactic.

        This is the point of Part B: a tactic pair accumulates evidence from every
        technique pair beneath it, so the statistic is supported where the
        technique-level one is not.
        """
        n_tactics = len(TACTIC_ORDER) + 1
        counts = np.zeros((n_tactics, n_tactics), dtype=np.float64)
        for prefix, nxt, dup in zip(examples["prefix"], examples["next"],
                                    examples["dup_count"].to_numpy(dtype=float)):
            source = multi_hot([prefix[-1]], self.mapping)[0]
            target = multi_hot([str(nxt)], self.mapping)[0]
            counts += dup * np.outer(source, target)
        self.tactic_counts = counts
        totals = counts.sum(axis=1, keepdims=True)
        self.tactic_probability = counts / np.where(totals > 0, totals, 1.0)
        # Project onto candidates: mean over each candidate's own tactics.
        per_candidate = self.cand_tactics.sum(axis=1, keepdims=True)
        per_candidate[per_candidate == 0] = 1.0
        self.tactic_prob_to_cand = self.tactic_probability @ self.cand_tactics.T / per_candidate.T
        self.tactic_count_to_cand = self.tactic_counts @ self.cand_tactics.T / per_candidate.T

    # ---- transform ---------------------------------------------------------
    def transform(self, examples: pd.DataFrame):
        X_base, qid, n_cand = self.base.transform(examples)
        blocks = [X_base.reshape(len(examples), n_cand, len(BASE_FEATURES))]
        if self.use_graph:
            blocks.append(self._graph_block(examples, n_cand))
        if self.use_tactics:
            blocks.append(self._tactic_block(examples, n_cand))
        if self.use_interactions:
            blocks.append(self._interaction_block(examples, n_cand))
        X = np.concatenate(blocks, axis=-1).reshape(len(examples) * n_cand, -1)
        return np.ascontiguousarray(X, dtype=np.float32), qid, n_cand

    def _graph_block(self, examples, n_cand):
        out = np.zeros((len(examples), n_cand, len(GRAPH_FEATURES)), dtype=np.float32)
        out[:, :, 7:] = self.cand_embed                      # continuous candidate identity
        graph = self.graph
        for row, prefix in enumerate(examples["prefix"]):
            member_rows = [graph.index[t] for t in dict.fromkeys(prefix) if t in graph.index]
            last_row = graph.index.get(prefix[-1], -1)
            if last_row >= 0:
                out[row, :, 0] = self.cand_unit @ graph.unit[last_row]
                out[row, :, 3] = graph.hops[last_row][self.cand_graph_row.clip(min=0)]
                out[row, :, 5] = graph.path_cost[last_row][self.cand_graph_row.clip(min=0)]
            if member_rows:
                centroid = graph.unit[member_rows].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    out[row, :, 1] = self.cand_unit @ (centroid / norm)
                sims = self.cand_unit @ graph.unit[member_rows].T
                out[row, :, 2] = sims.max(axis=1)
                out[row, :, 4] = graph.hops[np.ix_(member_rows, self.cand_graph_row.clip(min=0))].min(axis=0)
                out[row, :, 6] = graph.path_cost[np.ix_(member_rows, self.cand_graph_row.clip(min=0))].min(axis=0)
            out[row, ~self.cand_valid, 3] = out[row, ~self.cand_valid, 4] = 8.0
            out[row, ~self.cand_valid, 5] = out[row, ~self.cand_valid, 6] = 20.0
        return out

    def _tactic_block(self, examples, n_cand):
        n_t = len(tactic_columns())
        out = np.zeros((len(examples), n_cand, len(TACTIC_FEATURES)), dtype=np.float32)
        out[:, :, :n_t] = self.cand_tactics
        cand_first_index = np.full(n_cand, -1.0, dtype=np.float32)
        for j in range(n_cand):
            hits = np.flatnonzero(self.cand_tactics[j, :len(TACTIC_ORDER)] > 0)
            if len(hits):
                cand_first_index[j] = float(hits.min())
        for row, prefix in enumerate(examples["prefix"]):
            last_tactics = multi_hot([prefix[-1]], self.mapping)[0]
            active = np.flatnonzero(last_tactics > 0)
            if len(active):
                out[row, :, n_t + 0] = self.tactic_prob_to_cand[active].mean(axis=0)
                out[row, :, n_t + 1] = self.tactic_count_to_cand[active].mean(axis=0)
            prefix_tactics = multi_hot(list(dict.fromkeys(prefix)), self.mapping).max(axis=0)
            out[row, :, n_t + 2] = self.cand_tactics @ prefix_tactics
            out[row, :, n_t + 3] = (out[row, :, n_t + 2] == 0).astype(np.float32)
            reached = np.flatnonzero(prefix_tactics[:len(TACTIC_ORDER)] > 0)
            furthest = float(reached.max()) if len(reached) else -1.0
            out[row, :, n_t + 4] = cand_first_index - furthest
        return out

    def _interaction_block(self, examples, n_cand):
        out = np.zeros((len(examples), n_cand, len(INTERACTION_FEATURES)), dtype=np.float32)
        base = self.base
        n_decay = len(DECAY_LAMBDAS)
        for row, prefix in enumerate(examples["prefix"]):
            prefix = tuple(prefix)
            length = len(prefix)
            nearest = np.zeros(n_cand, dtype=np.float32)
            contributors = np.zeros(n_cand, dtype=np.float32)
            lift_recency = np.zeros(n_cand, dtype=np.float32)
            for position, item in enumerate(prefix):
                distance = length - 1 - position          # 0 == most recent
                probabilities = base.trans1_norm.get(item)
                if probabilities:
                    for technique, probability in probabilities.items():
                        j = base.cand_index.get(technique)
                        if j is None:
                            continue
                        for d, decay in enumerate(DECAY_LAMBDAS):
                            out[row, j, d] += (decay ** distance) * probability
                        contributors[j] += 1.0
                        # Keep the most recent contributing position, not the first.
                        nearest[j] = max(nearest[j], float(length - distance))
                partners = base.pair_count.get(item)
                count_a = base.item_count.get(item, 0.0)
                if partners and count_a > 0:
                    weight = 0.8 ** distance
                    for partner, pair in partners.items():
                        j = base.cand_index.get(partner)
                        if j is None:
                            continue
                        count_b = base.item_count.get(partner, 0.0)
                        if count_b > 0:
                            lift_recency[j] += weight * pair * base.n_transactions / (count_a * count_b)
            out[row, :, n_decay + 0] = nearest
            out[row, :, n_decay + 1] = contributors
            out[row, :, n_decay + 2] = contributors / max(length, 1)
            out[row, :, n_decay + 3] = lift_recency
        return out

    def labels(self, examples: pd.DataFrame) -> np.ndarray:
        return self.base.labels(examples)


class PipelineRankerAdapter:
    """XGBRanker over the assembled pipeline features; plugs into the CV harness."""

    def __init__(self, name: str = "Pipeline", n_estimators: int = 500,
                 learning_rate: float = 0.06, max_depth: int = 7,
                 min_child_weight: float = 2.0, subsample: float = 0.9,
                 colsample_bytree: float = 0.8, **groups):
        self.name = name
        self.groups = groups
        self.params = {
            "objective": "rank:ndcg",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": "hist",
            "random_state": config.RANDOM_SEED,
            "n_jobs": 1,
            "verbosity": 0,
        }

    def fit(self, examples: pd.DataFrame):
        import xgboost as xgb

        self.builder = PipelineFeatureBuilder(**self.groups).fit(examples)
        X, qid, _ = self.builder.transform(examples)
        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X, self.builder.labels(examples), qid=qid)
        return self

    def rank(self, examples: pd.DataFrame):
        X, _, n_cand = self.builder.transform(examples)
        scores = self.model.predict(X).reshape(len(examples), n_cand)
        names = np.asarray(self.builder.candidates, dtype=object)
        return [names[order].tolist()
                for order in np.argsort(-scores, axis=1, kind="stable")]

    def importance(self) -> pd.DataFrame:
        names = self.builder.feature_names
        gains = self.model.get_booster().get_score(importance_type="gain")
        rows = [{"feature": names[int(k[1:])], "gain": v} for k, v in gains.items()]
        return pd.DataFrame(rows).sort_values("gain", ascending=False)


def main():
    from cv import assert_test_sealed, load_development_examples
    from cv_harness import comparison_table, run_cv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="all folds + paired CIs")
    args = parser.parse_args()

    assert_test_sealed()
    examples = load_development_examples()
    out_dir = config.RESULTS_DIR / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Development set: {len(examples):,} examples / "
          f"{examples['seq_id'].nunique():,} sequences. Test split not loaded.\n")

    variants = {
        "base_mining_only": dict(use_graph=False, use_tactics=False, use_interactions=False),
        "A_graph": dict(use_graph=True, use_tactics=False, use_interactions=False),
        "A+B_tactics": dict(use_graph=True, use_tactics=True, use_interactions=False),
        "A+C_interactions": dict(use_graph=True, use_tactics=False, use_interactions=True),
        "A+B+C_full": dict(use_graph=True, use_tactics=True, use_interactions=True),
    }
    max_folds = None if args.full else 1
    folds, seqs, pooled = [], {}, []
    for name, groups in variants.items():
        print(f"{name}:")
        f, s, p = run_cv(lambda g=groups, n=name: PipelineRankerAdapter(name=n, **g),
                         name, examples, max_folds=max_folds)
        folds.append(f); seqs[name] = s; pooled.append(p)

    table = pd.DataFrame(pooled)[
        ["model", "seq_top_1", "seq_top_3", "seq_top_5", "seq_mrr",
         "seq_macro_recall", "seq_macro_f1"]].round(4)
    print("\n=== Pipeline ablation (D masking always on, XGBRanker head) ===")
    print(table.to_string(index=False))
    table.to_csv(out_dir / ("pipeline_cv_summary.csv" if args.full
                            else "pipeline_fold0_summary.csv"), index=False)
    pd.concat(folds, ignore_index=True).to_csv(
        out_dir / ("pipeline_fold_metrics.csv" if args.full
                   else "pipeline_fold0_metrics.csv"), index=False)
    print("\nreference, same protocol: LightGBM softprob 5-fold seq_mrr=0.5386 "
          "(fold 0 = 0.5534)")

    if args.full and len(seqs) > 1:
        pairs = comparison_table(seqs)
        print("\n=== Paired comparisons between variants (per-sequence MRR) ===")
        print(pairs[["model_a", "model_b", "mean_diff", "ci_2.5", "ci_97.5",
                     "significant"]].to_string(index=False))
        pairs.to_csv(out_dir / "pipeline_paired.csv", index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
