"""Does ATT&CK tactic abstraction improve next-technique ranking?

Run from the repository root:
  python src/eval/tactic_experiments.py            # fold 0 screen
  python src/eval/tactic_experiments.py --full     # 5-fold with paired CIs

Why tactics might help where mined patterns did not
---------------------------------------------------
Mined-pattern features are deterministic functions of the prefix, and the 511
`seen_*`/`last-k` features already determine the prefix completely, so they
carried no new information (measured: -0.0059). The tactic taxonomy is
different: it is external human knowledge that is *not derivable* from the
sightings data. It also targets the measured bottleneck, class sparsity -- about
600 examples per tactic against 89 per technique, with 47 of 94 techniques
holding fewer than 5 examples.

Three designs are tested, weakest to strongest:

A. `prefix_tactics`   - tactic multi-hot of the prefix added to the softprob
                        model. Prefix-level, so valid as extra columns.
B. `hierarchical`     - 14 binary tactic heads predict the next technique's
                        tactic, then reweight the technique distribution:
                        score(c) = P_technique(c) * tactic_score(c)^lambda.
                        This is where class sparsity actually bites, because it
                        puts the external knowledge on the output side.
C. `ranker_tactics`   - candidate tactic multi-hot plus tactic-transition
                        evidence in the candidate ranker, where candidate-side
                        features are natural.

lambda = 0 recovers the baseline exactly and is kept in the sweep as a sanity
check. The test split is never loaded.
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
from cv import assert_test_sealed, development_folds, load_development_examples
from cv_harness import aggregate, paired_compare, per_sequence, score_rankings
from lightgbm_predictor import (_base_params, _class_names, _make_features,
                                _ranked_labels, _seen_mask, _sequence_weights)
from tactic_map import TACTIC_ORDER, load_mapping, multi_hot, tactic_columns

OUT_DIR = config.RESULTS_DIR / "tactics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LGB_GRID9 = {"num_leaves": 31, "min_child_samples": 10, "reg_lambda": 1.0}
LGB_ROUNDS = 101
N_TACTIC_COLS = len(TACTIC_ORDER) + 1


def _lgb(binary: bool = False, rounds: int = LGB_ROUNDS):
    import lightgbm as lgb
    params = _base_params(dict(LGB_GRID9), rounds)
    params["n_jobs"] = 1
    if binary:
        params["objective"] = "binary"
    return lgb.LGBMClassifier(**params)


def prefix_tactic_block(examples: pd.DataFrame, mapping) -> pd.DataFrame:
    """Prefix-level tactic features: which tactics the prefix has touched.

    Three views, because they answer different questions: the union says which
    tactics appear anywhere in the prefix, the last-technique block says where
    the campaign currently is, and the scalars summarise breadth and how far
    along the kill chain it has reached.
    """
    index = {t: i for i, t in enumerate(TACTIC_ORDER)}
    union = np.zeros((len(examples), N_TACTIC_COLS), dtype=np.float32)
    last = np.zeros((len(examples), N_TACTIC_COLS), dtype=np.float32)
    scalars = np.zeros((len(examples), 2), dtype=np.float32)
    for row, prefix in enumerate(examples["prefix"]):
        reached = []
        for technique in set(prefix):
            tactics = mapping.get(str(technique))
            if not tactics:
                union[row, -1] = 1.0
                continue
            for tactic in tactics:
                j = index.get(tactic)
                if j is not None:
                    union[row, j] = 1.0
                    reached.append(j)
        last[row] = multi_hot([prefix[-1]], mapping)[0]
        scalars[row, 0] = union[row, :len(TACTIC_ORDER)].sum()
        # Furthest point reached in kill-chain order; a coarse progress signal.
        scalars[row, 1] = max(reached) if reached else -1.0
    columns = ([f"pfx_{c}" for c in tactic_columns()]
               + [f"last_{c}" for c in tactic_columns()]
               + ["pfx_n_tactics", "pfx_max_tactic_index"])
    return pd.DataFrame(np.hstack([union, last, scalars]), columns=columns)


def tactic_scores(fit: pd.DataFrame, score: pd.DataFrame, mapping, candidates):
    """P(next technique belongs to tactic t | prefix) from 14+1 binary heads.

    A binary head per tactic is far better supported than a 94-way head: every
    tactic has hundreds of positive examples. Heads with no positives in a fold
    fall back to that tactic's base rate.
    """
    techniques = sorted(set(fit["next"].astype(str)) | {t for p in fit["prefix"] for t in p})
    X_fit = _make_features(fit, techniques)
    X_score = _make_features(score, techniques)
    weights = _sequence_weights(fit)
    target = multi_hot(fit["next"].astype(str).tolist(), mapping)
    predicted = np.zeros((len(score), N_TACTIC_COLS), dtype=np.float64)
    for j in range(N_TACTIC_COLS):
        y = target[:, j].astype(int)
        rate = float(y.mean())
        if y.sum() < 2 or rate in (0.0, 1.0):
            predicted[:, j] = rate
            continue
        head = _lgb(binary=True)
        head.fit(X_fit, y, sample_weight=weights)
        positive = int(np.flatnonzero(head.classes_ == 1)[0])
        predicted[:, j] = head.predict_proba(X_score)[:, positive]
    # Map each candidate to the mean predicted probability of its own tactics.
    candidate_tactics = multi_hot(list(candidates), mapping)
    counts = candidate_tactics.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1.0
    return predicted @ candidate_tactics.T / counts.T


def run_fold(fit: pd.DataFrame, score: pd.DataFrame, mapping, lambdas):
    """All three designs on one fold. Returns metric rows and per-sequence scores."""
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder().fit(fit["next"].astype(str))
    y = encoder.transform(fit["next"].astype(str))
    techniques = sorted(set(fit["next"].astype(str)) | {t for p in fit["prefix"] for t in p})
    X_fit, X_score = _make_features(fit, techniques), _make_features(score, techniques)
    weights = _sequence_weights(fit)
    names = np.asarray(encoder.classes_, dtype=object)

    rows, seq_scores = [], {}

    def record(tag, probabilities, class_names, features):
        ranked = _ranked_labels(probabilities, class_names, _seen_mask(features, class_names))
        scored = score_rankings([r.tolist() for r in ranked], score)
        rows.append({"model": tag, **{k: round(v, 4) for k, v in aggregate(scored).items()}})
        seq_scores[tag] = per_sequence(scored)
        return scored

    # --- baseline -----------------------------------------------------------
    base = _lgb()
    base.fit(X_fit, y, sample_weight=weights)
    class_names = _class_names(base.classes_, names)
    base_probabilities = base.predict_proba(X_score)
    record("baseline_511", base_probabilities, class_names, X_score)

    # --- A: prefix tactic features -----------------------------------------
    T_fit, T_score = prefix_tactic_block(fit, mapping), prefix_tactic_block(score, mapping)
    augmented_fit = pd.concat([X_fit, T_fit], axis=1)
    augmented_score = pd.concat([X_score, T_score], axis=1)
    model_a = _lgb()
    model_a.fit(augmented_fit, y, sample_weight=weights)
    record(f"A_prefix_tactics_+{T_fit.shape[1]}f",
           model_a.predict_proba(augmented_score),
           _class_names(model_a.classes_, names), augmented_score)

    # --- B: hierarchical tactic reweighting --------------------------------
    candidate_scores = tactic_scores(fit, score, mapping, class_names)
    for lam in lambdas:
        weighted = base_probabilities * np.power(np.clip(candidate_scores, 1e-9, None), lam)
        record(f"B_hierarchical_lambda{lam}", weighted, class_names, X_score)

    return pd.DataFrame(rows), seq_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="run all folds and report paired CIs")
    parser.add_argument("--lambdas", type=float, nargs="*",
                        default=[0.0, 0.25, 0.5, 1.0, 2.0])
    args = parser.parse_args()

    assert_test_sealed()
    examples = load_development_examples()
    mapping = load_mapping()
    vocabulary = sorted(set(examples["next"].astype(str))
                        | {t for p in examples["prefix"] for t in p})
    covered = sum(1 for t in vocabulary if mapping.get(t))
    print(f"Development set: {len(examples):,} examples / "
          f"{examples['seq_id'].nunique():,} sequences. Test split not loaded.")
    print(f"Tactic mapping covers {covered}/{len(vocabulary)} techniques "
          f"({covered/len(vocabulary):.1%}); unmapped get an explicit flag.\n")

    all_rows, fold_seq = [], defaultdict(list)
    for fold, fit_idx, score_idx in development_folds(examples):
        fit = examples.iloc[fit_idx].reset_index(drop=True)
        score = examples.iloc[score_idx].reset_index(drop=True)
        rows, seq_scores = run_fold(fit, score, mapping, args.lambdas)
        rows.insert(1, "fold", fold)
        all_rows.append(rows)
        for tag, frame in seq_scores.items():
            fold_seq[tag].append(frame)
        print(f"--- fold {fold} ---")
        print(rows[["model", "seq_top_1", "seq_top_5", "seq_mrr", "seq_macro_recall"]]
              .to_string(index=False))
        if not args.full:
            break

    folds = pd.concat(all_rows, ignore_index=True)
    folds.to_csv(OUT_DIR / "tactic_fold_metrics.csv", index=False)

    if args.full:
        pooled = (folds.groupby("model")[["seq_top_1", "seq_top_3", "seq_top_5",
                                          "seq_mrr", "seq_macro_recall", "seq_macro_f1"]]
                  .mean().round(4))
        pooled["seq_mrr_sd"] = folds.groupby("model")["seq_mrr"].std(ddof=1).round(4)
        print("\n=== 5-fold means ===")
        print(pooled.to_string())
        pooled.to_csv(OUT_DIR / "tactic_cv_summary.csv")

        combined = {tag: pd.concat(frames).sort_index() for tag, frames in fold_seq.items()}
        reference = combined["baseline_511"]
        comparisons = [
            paired_compare(combined[tag], reference, tag, "baseline_511")
            for tag in combined if tag != "baseline_511"
        ]
        table = pd.DataFrame(comparisons)
        print("\n=== Paired against baseline (per-sequence MRR, bootstrap over sequences) ===")
        print(table[["model_a", "mean_diff", "ci_2.5", "ci_97.5", "significant"]]
              .to_string(index=False))
        table.to_csv(OUT_DIR / "tactic_paired_vs_baseline.csv", index=False)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
