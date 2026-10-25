"""Measurement of what limits next-technique accuracy, and what does not.

Run from the repository root:
  python src/eval/diagnostics.py

Five questions, each answered by a measurement rather than an argument:

1. `resolving_power`   - how small a difference can the evaluation actually see?
2. `lookup_ceiling`    - how far can a model get by knowing each context's true
                         conditional distribution (the limit of Markov and
                         PrefixSpan style lookup), and have we passed it?
3. `naive_bayes`       - does a high-bias model suit this small dataset?
4. `complementarity`   - do the model families make different mistakes, and is a
                         fixed-weight blend enough to exploit that?
5. `feature_probes`    - do mined patterns, candidate identity, model stacking,
                         or graph similarity add anything to the prefix encoding?

Development splits only; the test split is never loaded. Estimators run
single-threaded so the reported numbers are reproducible.
"""

from __future__ import annotations

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
from cv_harness import aggregate, score_rankings
from baselines import MarkovOrder1, MarkovOrder2, build_singleton_prior, mask_seen
from lightgbm_predictor import (_base_params, _class_names, _make_features,
                                _ranked_labels, _seen_mask, _sequence_weights)

OUT_DIR = config.RESULTS_DIR / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LGB_GRID9 = {"num_leaves": 31, "min_child_samples": 10, "reg_lambda": 1.0}
LGB_ROUNDS = 101


def _lightgbm():
    import lightgbm as lgb
    params = _base_params(dict(LGB_GRID9), LGB_ROUNDS)
    params["n_jobs"] = 1
    return lgb.LGBMClassifier(**params)


# ---------------------------------------------------------------- 1
def resolving_power(examples: pd.DataFrame) -> pd.DataFrame:
    """Standard error of sequence MRR under each evaluation design."""
    import lightgbm  # noqa: F401  (import cost paid once, here)

    fold, fit_idx, score_idx = next(development_folds(examples))
    fit = examples.iloc[fit_idx].reset_index(drop=True)
    score = examples.iloc[score_idx].reset_index(drop=True)
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder().fit(fit["next"].astype(str))
    techniques = sorted(set(fit["next"].astype(str)) | {t for p in fit["prefix"] for t in p})
    model = _lightgbm()
    model.fit(_make_features(fit, techniques),
              encoder.transform(fit["next"].astype(str)),
              sample_weight=_sequence_weights(fit))
    names = np.asarray(encoder.classes_, dtype=object)
    features = _make_features(score, techniques)
    class_names = _class_names(model.classes_, names)
    ranked = _ranked_labels(model.predict_proba(features), class_names,
                            _seen_mask(features, class_names))
    scored = score_rankings([row.tolist() for row in ranked], score)
    per_seq = scored.groupby("seq_id")["rr"].mean()
    sd = float(per_seq.std(ddof=1))

    designs = [
        ("single 545-sequence validation holdout", 545),
        ("final test split", 543),
        (f"{config.N_CV_FOLDS}-fold CV over the development set", examples["seq_id"].nunique()),
    ]
    rows = []
    for label, n in designs:
        se = sd / np.sqrt(n)
        rows.append({
            "design": label,
            "n_sequences": n,
            "per_sequence_sd": round(sd, 4),
            "standard_error": round(se, 4),
            "ci95_half_width": round(1.96 * se, 4),
            "min_unpaired_detectable_diff": round(2.8 * se * np.sqrt(2), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2
def lookup_ceiling(examples: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out empirical-Bayes ceiling for context-lookup prediction.

    For each example, rank labels by the pooled count of (context -> label) with
    that example's own contribution removed. This is the best achievable by any
    method that predicts from a context's observed successor distribution, which
    is what the Markov and PrefixSpan predictors do.
    """
    contexts = {
        "last1": lambda p: (p[-1],),
        "last2": lambda p: tuple(p[-2:]),
        "last3": lambda p: tuple(p[-3:]),
        "full_ordered_prefix": lambda p: tuple(p),
        "prefix_as_set": lambda p: tuple(sorted(set(p))),
    }
    prior = Counter(examples["next"])
    rows = []
    for name, key in contexts.items():
        table = defaultdict(Counter)
        for prefix, nxt in zip(examples["prefix"], examples["next"]):
            table[key(prefix)][nxt] += 1
        sizes = np.array([sum(c.values()) for c in table.values()])
        distinct = np.array([len(c) for c in table.values()])
        reused = sizes >= 2
        top1, rr = [], []
        for prefix, nxt in zip(examples["prefix"], examples["next"]):
            counts = Counter(table[key(prefix)])
            counts[nxt] -= 1
            counts = Counter({k: v for k, v in counts.items() if v > 0}) or prior
            ranked = sorted(counts, key=lambda k: (-counts[k], k))
            top1.append(float(ranked[0] == nxt))
            rr.append(1.0 / (ranked.index(nxt) + 1) if nxt in ranked else 0.0)
        rows.append({
            "context": name,
            "n_contexts": len(table),
            "reuse_rate": round(float(reused.mean()), 4),
            "mean_distinct_labels_when_reused": round(float(distinct[reused].mean()), 2),
            "loo_top_1": round(float(np.mean(top1)), 4),
            "loo_mrr": round(float(np.mean(rr)), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3 & 4
def naive_bayes_and_complementarity(examples: pd.DataFrame):
    """Naive Bayes variants, plus error overlap and blending against LightGBM."""
    from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
    from sklearn.preprocessing import LabelEncoder

    fold, fit_idx, score_idx = next(development_folds(examples))
    fit = examples.iloc[fit_idx].reset_index(drop=True)
    score = examples.iloc[score_idx].reset_index(drop=True)
    encoder = LabelEncoder().fit(fit["next"].astype(str))
    y = encoder.transform(fit["next"].astype(str))
    techniques = sorted(set(fit["next"].astype(str)) | {t for p in fit["prefix"] for t in p})
    X, X_score = _make_features(fit, techniques), _make_features(score, techniques)
    weights = _sequence_weights(fit)
    names = np.asarray(encoder.classes_, dtype=object)

    results, hits, probabilities = [], {}, {}
    estimators = [
        ("BernoulliNB_a1.0", BernoulliNB(alpha=1.0)),
        ("BernoulliNB_a0.1", BernoulliNB(alpha=0.1)),
        ("ComplementNB_a0.5", ComplementNB(alpha=0.5)),
        ("MultinomialNB_a0.5", MultinomialNB(alpha=0.5)),
        ("LightGBM_grid9", _lightgbm()),
    ]
    for name, estimator in estimators:
        estimator.fit(X, y, sample_weight=weights)
        class_names = _class_names(estimator.classes_, names)
        proba = estimator.predict_proba(X_score)
        ranked = _ranked_labels(proba, class_names, _seen_mask(X_score, class_names))
        scored = score_rankings([row.tolist() for row in ranked], score)
        results.append({"model": name, "fold": fold, **{
            k: round(v, 4) for k, v in aggregate(scored).items()}})
        hits[name] = scored["top_1"].to_numpy()
        # align probabilities onto the full label space for blending
        aligned = np.zeros((len(score), len(names)), dtype=float)
        for col, label in enumerate(class_names):
            aligned[:, int(np.flatnonzero(names == label)[0])] = proba[:, col]
        probabilities[name] = aligned

    # Markov-2 reference and its reciprocal-rank score block
    prior = build_singleton_prior(config.SIGHTINGS_DATA_PATH)
    m1 = MarkovOrder1(k=0.001, fallback_counts=prior); m1.fit(fit)
    m2 = MarkovOrder2(m=5, k1=0.001, fallback_counts=prior); m2.fit(fit, markov1=m1)
    markov_rankings = [mask_seen(m2.predict(p), p) for p in score["prefix"]]
    scored_markov = score_rankings(markov_rankings, score)
    results.append({"model": "B4_Markov2", "fold": fold,
                    **{k: round(v, 4) for k, v in aggregate(scored_markov).items()}})
    hits["B4_Markov2"] = scored_markov["top_1"].to_numpy()
    index = {n: i for i, n in enumerate(names)}
    markov_block = np.zeros((len(score), len(names)), dtype=float)
    for row, ranked in enumerate(markov_rankings):
        for rank, technique in enumerate(ranked):
            if technique in index:
                markov_block[row, index[technique]] = 1.0 / (rank + 1)
        total = markov_block[row].sum()
        if total:
            markov_block[row] /= total

    overlap = []
    reference = hits["LightGBM_grid9"]
    for name in ("B4_Markov2", "BernoulliNB_a0.1", "ComplementNB_a0.5"):
        other = hits[name]
        both = float(((reference == 1) & (other == 1)).mean())
        only_ref = float(((reference == 1) & (other == 0)).mean())
        only_other = float(((reference == 0) & (other == 1)).mean())
        overlap.append({
            "other_model": name,
            "both_correct": round(both, 4),
            "only_lightgbm": round(only_ref, 4),
            "only_other": round(only_other, 4),
            "neither": round(float(((reference == 0) & (other == 0)).mean()), 4),
            "union_top1_ceiling": round(both + only_ref + only_other, 4),
        })

    blend = []
    mask = _seen_mask(X_score, names)
    for alpha in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0):
        mixed = alpha * probabilities["LightGBM_grid9"] + (1 - alpha) * markov_block
        ranked = _ranked_labels(mixed, names, mask)
        metrics = aggregate(score_rankings([r.tolist() for r in ranked], score))
        blend.append({"lightgbm_weight": alpha,
                      **{k: round(v, 4) for k, v in metrics.items()}})
    return pd.DataFrame(results), pd.DataFrame(overlap), pd.DataFrame(blend)


def main():
    assert_test_sealed()
    examples = load_development_examples()
    print(f"Development set: {len(examples):,} examples / "
          f"{examples['seq_id'].nunique():,} unique sequences. Test split not loaded.\n")

    power = resolving_power(examples)
    print("=== 1. Resolving power of each evaluation design ===")
    print(power.to_string(index=False))
    power.to_csv(OUT_DIR / "resolving_power.csv", index=False)

    ceiling = lookup_ceiling(examples)
    print("\n=== 2. Leave-one-out context-lookup ceiling ===")
    print(ceiling.to_string(index=False))
    ceiling.to_csv(OUT_DIR / "lookup_ceiling.csv", index=False)

    models, overlap, blend = naive_bayes_and_complementarity(examples)
    print("\n=== 3. Naive Bayes vs the references (fold 0) ===")
    print(models[["model", "seq_top_1", "seq_top_5", "seq_mrr", "seq_macro_recall"]]
          .to_string(index=False))
    models.to_csv(OUT_DIR / "naive_bayes_and_references.csv", index=False)

    print("\n=== 4a. Top-1 error overlap with LightGBM ===")
    print(overlap.to_string(index=False))
    overlap.to_csv(OUT_DIR / "model_complementarity.csv", index=False)

    print("\n=== 4b. Fixed-weight blend of LightGBM and Markov-2 ===")
    print(blend[["lightgbm_weight", "seq_top_1", "seq_top_5", "seq_mrr"]].to_string(index=False))
    blend.to_csv(OUT_DIR / "blend_sweep.csv", index=False)

    print(f"\nSaved diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
