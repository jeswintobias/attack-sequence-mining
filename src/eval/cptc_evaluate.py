"""Run the project's predictors on CPTC-2018 as a second, timestamped dataset.

Run from the repository root:
  python src/eval/cptc_evaluate.py

Purpose
-------
Every result so far comes from one dataset whose ordering is *reported* order.
CPTC-2018 has real microsecond timestamps and six independent teams, so it tests
two things nothing else in this project can:

1. Whether the method generalises to a different dataset and label space.
2. Whether our numbers are competitive with published work. arXiv 2408.09888
   evaluates the same task on this data and reports top-3 accuracy of 67.27%
   (best method, rSPDFA with hybrid traversal) against 62.33% for its strongest
   baseline, over 148 symbols.

Important differences from the sightings pipeline, both deliberate:

- **Prefix masking is disabled.** 88-93% of CPTC targets repeat a symbol already
  in the prefix, because revisiting an attack stage later in a campaign is real
  behaviour. The sightings pipeline masks repeats only because DEC-03's
  first-occurrence deduplication makes them impossible there.
- **Folds are built here**, grouped by trace and stratified by length, rather
  than reusing the sightings development folds.

Comparability caveat: the episode segmentation is a fixed-window simplification
of SAGE's frequency-peak method, so trace counts are close (640-786 against 555)
but mean lengths are longer. Treat the published figures as context, not as a
head-to-head benchmark.
"""

from __future__ import annotations

import sys
from collections import Counter
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

# Repeats are legitimate in this dataset; see the module docstring.
config.MASK_SEEN_TECHNIQUES = False

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from cv_harness import aggregate, paired_compare, per_sequence, score_rankings
from baselines import MajorityBaseline, MarkovOrder1, MarkovOrder2
from lightgbm_predictor import _base_params, _class_names, _make_features

OUT_DIR = config.RESULTS_DIR / "cptc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER = {"best_top_3": 0.6727, "baseline_top_3": 0.6233, "symbols": 148,
         "traces": 555, "avg_length": 5.78}


def _folds(examples: pd.DataFrame, n_splits: int = 5):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=config.RANDOM_SEED)
    groups = examples["seq_id"].to_numpy()
    strata = examples["length_bucket"].to_numpy()
    yield from enumerate(splitter.split(examples, strata, groups))


def _rank_markov(model, examples):
    return [model.predict(prefix) for prefix in examples["prefix"]]


def _rank_softprob(fit, score):
    import lightgbm as lgb

    encoder = LabelEncoder().fit(fit["next"].astype(str))
    y = encoder.transform(fit["next"].astype(str))
    symbols = sorted(set(fit["next"].astype(str)) | {s for p in fit["prefix"] for s in p})
    params = _base_params({"num_leaves": 31, "min_child_samples": 10, "reg_lambda": 1.0}, 101)
    params["n_jobs"] = 1
    model = lgb.LGBMClassifier(**params)
    model.fit(_make_features(fit, symbols), y,
              sample_weight=np.ones(len(fit), dtype=float))
    probabilities = model.predict_proba(_make_features(score, symbols))
    names = _class_names(model.classes_, np.asarray(encoder.classes_, dtype=object))
    order = np.argsort(-probabilities, axis=1, kind="stable")
    return [names[row].tolist() for row in order]


def evaluate(examples: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    fold_rows, seq_scores = [], {name: [] for name in
                                 ("B1_Majority", "B3_Markov1", "B4_Markov2", "LightGBM_softprob")}
    for fold, (fit_idx, score_idx) in _folds(examples):
        fit = examples.iloc[fit_idx].reset_index(drop=True)
        score = examples.iloc[score_idx].reset_index(drop=True)
        prior = Counter(fit["next"].astype(str))

        majority = MajorityBaseline()
        majority.fit(fit)
        m1 = MarkovOrder1(k=0.001, fallback_counts=prior); m1.fit(fit)
        m2 = MarkovOrder2(m=5, k1=0.001, fallback_counts=prior); m2.fit(fit, markov1=m1)

        rankings = {
            "B1_Majority": [majority.predict(None)] * len(score),
            "B3_Markov1": _rank_markov(m1, score),
            "B4_Markov2": _rank_markov(m2, score),
            "LightGBM_softprob": _rank_softprob(fit, score),
        }
        for name, ranked in rankings.items():
            scored = score_rankings(ranked, score)
            metrics = aggregate(scored)
            fold_rows.append({"dataset": label, "model": name, "fold": fold,
                              **{k: round(v, 4) for k, v in metrics.items()}})
            seq_scores[name].append(per_sequence(scored))
        print(f"  fold {fold}: " + "  ".join(
            f"{n.split('_')[0]}={aggregate(score_rankings(r, score))['seq_mrr']:.3f}"
            for n, r in rankings.items()), flush=True)

    folds = pd.DataFrame(fold_rows)
    combined = {name: pd.concat(frames).sort_index()
                for name, frames in seq_scores.items() if frames}
    return folds, combined


def main():
    print("=" * 78)
    print("CPTC-2018 EVALUATION  (prefix masking disabled; repeats are real here)")
    print("=" * 78)
    results, all_seq = [], {}
    for suffix, label in (("stage", "stage only"), ("service", "stage|service")):
        path = config.PROCESSED_DATA_DIR / f"cptc_examples_{suffix}.pkl"
        if not path.exists():
            print(f"\n{label}: {path} missing; run python src/core/cptc_loader.py")
            continue
        examples = pd.read_pickle(path)
        print(f"\n[{label}] {len(examples):,} examples / "
              f"{examples['seq_id'].nunique():,} traces / "
              f"{examples['next'].nunique()} symbols")
        folds, combined = evaluate(examples, label)
        results.append(folds)
        all_seq[label] = combined

        summary = (folds.groupby("model")[["seq_top_1", "seq_top_3", "seq_top_5", "seq_mrr"]]
                   .mean().round(4))
        summary["seq_mrr_sd"] = folds.groupby("model")["seq_mrr"].std(ddof=1).round(4)
        print(f"\n  5-fold means, {label}:")
        print(summary.to_string())

    if results:
        everything = pd.concat(results, ignore_index=True)
        everything.to_csv(OUT_DIR / "cptc_fold_metrics.csv", index=False)
        pooled = (everything.groupby(["dataset", "model"])
                  [["seq_top_1", "seq_top_3", "seq_top_5", "seq_mrr"]].mean().round(4))
        pooled.to_csv(OUT_DIR / "cptc_summary.csv")

        print("\n" + "=" * 78)
        print("CONTEXT: published results on this dataset (arXiv 2408.09888)")
        print("=" * 78)
        print(f"  their best method   top-3 = {PAPER['best_top_3']:.4f}  "
              f"({PAPER['symbols']} symbols, {PAPER['traces']} traces, "
              f"avg length {PAPER['avg_length']})")
        print(f"  their best baseline top-3 = {PAPER['baseline_top_3']:.4f}")
        for (dataset, model), row in pooled.iterrows():
            if model == "LightGBM_softprob":
                print(f"  ours, {dataset:<14} top-3 = {row['seq_top_3']:.4f}")
        print("\n  Not a strict head-to-head: episode segmentation and prefix"
              "\n  definitions differ. See module docstring.")

        # paired comparison within the richer label space
        if "stage|service" in all_seq:
            combined = all_seq["stage|service"]
            pairs = [paired_compare(combined["LightGBM_softprob"], combined[other],
                                    "LightGBM_softprob", other)
                     for other in ("B4_Markov2", "B3_Markov1", "B1_Majority")]
            table = pd.DataFrame(pairs)
            print("\n=== Paired per-trace MRR, stage|service ===")
            print(table[["model_a", "model_b", "mean_diff", "ci_2.5", "ci_97.5",
                         "significant"]].to_string(index=False))
            table.to_csv(OUT_DIR / "cptc_paired.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
