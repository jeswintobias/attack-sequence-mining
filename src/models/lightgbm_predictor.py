"""LightGBM next-technique experiments; train/validation only, never test.

Run from the repository root:
  python src/lightgbm_predictor.py --search grid
  python -m pip install optuna
  python src/lightgbm_predictor.py --search optuna
  python src/lightgbm_predictor.py --search halving

All predictors use prefix-presence, last-four-position, and prefix-length
features. Search CV groups by full sequence so prefixes from one sequence
cannot appear in both fit and scoring folds. Optuna uses early stopping on an
inner, sequence-grouped training holdout. HalvingSearchCV allocates n_estimators
as its resource, so early stopping is applied only to its final refit. The
external validation set is used to report the selected method; the test set is
never loaded. Optuna is optional; the halving option uses scikit-learn's
experimental HalvingRandomSearchCV API.
"""

from __future__ import annotations

import argparse
import json
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from baselines import evaluate_predictions
from evaluation_metrics import macro_recall_f1

MAX_ESTIMATORS = 400
EARLY_STOPPING_ROUNDS = 30


def _load_examples():
    processed = config.BASE_DIR / "data" / "processed"
    train = pd.read_pickle(processed / "train_examples.pkl")
    val = pd.read_pickle(processed / "val_examples.pkl")
    train_seq = pd.read_csv(processed / "train_sequences.csv")
    val_seq = pd.read_csv(processed / "val_sequences.csv")
    return train, val, train_seq, val_seq


def _make_features(examples: pd.DataFrame, techniques: list[str]) -> pd.DataFrame:
    """Encode unordered prefix membership plus recent ordered context."""
    tech_to_i = {t: i for i, t in enumerate(techniques)}
    X = np.zeros((len(examples), len(techniques) * 5 + 1), dtype=np.float32)
    for row_i, prefix in enumerate(examples["prefix"]):
        for t in set(prefix):
            if t in tech_to_i:
                X[row_i, tech_to_i[t]] = 1.0
        for offset in range(1, 5):
            if len(prefix) >= offset:
                t = prefix[-offset]
                if t in tech_to_i:
                    X[row_i, len(techniques) * offset + tech_to_i[t]] = 1.0
        X[row_i, -1] = len(prefix)
    cols = ([f"seen_{t}" for t in techniques]
            + [f"last{j}_{t}" for j in range(1, 5) for t in techniques]
            + ["prefix_length"])
    return pd.DataFrame(X, columns=cols)


def _sequence_weights(examples: pd.DataFrame) -> np.ndarray:
    """Each sequence's prefix rows sum to its duplicate count."""
    seq_ids = examples["full_seq"].map(lambda s: tuple(s))
    n_examples = seq_ids.map(seq_ids.value_counts())
    return examples["dup_count"].to_numpy(dtype=float) / n_examples.to_numpy(dtype=float)


def _groups(examples: pd.DataFrame) -> np.ndarray:
    """Stable integer group labels for full sequences."""
    return pd.factorize(examples["full_seq"].map(tuple), sort=True)[0]


def _base_params(params: dict, n_estimators: int = MAX_ESTIMATORS) -> dict:
    return {
        "objective": "multiclass",
        "metric": "None",
        "n_estimators": int(n_estimators),
        "learning_rate": 0.05,
        "max_depth": -1,
        "colsample_bytree": 0.8,
        "subsample": 0.9,
        "subsample_freq": 1,
        "random_state": config.RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": -1,
        **params,
    }


def _fit_with_group_early_stopping(X, y, weights, groups, params):
    """Find tree count on an inner sequence holdout, then refit all supplied rows."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.RANDOM_SEED)
    fit_idx, stop_idx = next(splitter.split(X, y, groups))
    # Early-stopping labels must have been represented in the fit fold.
    seen_labels = np.unique(y[fit_idx])
    stop_idx = stop_idx[np.isin(y[stop_idx], seen_labels)]
    if not len(stop_idx):
        raise ValueError("Inner early-stopping holdout has no labels seen in its fit fold")

    stop_groups = groups[stop_idx]
    fit_class_labels = np.unique(y[fit_idx])

    def sequence_mrr(y_true, y_pred):
        return (
            "sequence_mrr",
            _mrr_from_probabilities(y_true, y_pred, stop_groups, fit_class_labels),
            True,
        )

    stopping_model = lgb.LGBMClassifier(**_base_params(params))
    stopping_model.fit(
        X.iloc[fit_idx], y[fit_idx], sample_weight=weights[fit_idx],
        eval_X=X.iloc[stop_idx],
        eval_y=y[stop_idx],
        eval_metric=sequence_mrr,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iteration = int(stopping_model.best_iteration_ or MAX_ESTIMATORS)
    final_model = lgb.LGBMClassifier(**_base_params(params, best_iteration))
    final_model.fit(X, y, sample_weight=weights)
    return final_model, best_iteration


def _metrics(model, X, examples, labels):
    probabilities = model.predict_proba(X)
    output_labels = np.asarray(labels, dtype=object)
    records = []
    for i, (_, row) in enumerate(examples.iterrows()):
        rank = np.argsort(-probabilities[i], kind="stable")
        pred = output_labels[rank]
        score = evaluate_predictions(pred.tolist(), row["next"])
        records.append({
            **score,
            "actual": row["next"],
            "predicted": pred[0] if len(pred) else None,
            "full_seq": tuple(row["full_seq"]),
        })
    scored = pd.DataFrame(records)
    metrics = {}
    cols = ("top_1", "top_3", "top_5", "mrr")
    for col in cols:
        metrics[f"prefix_example_uw_{col}"] = float(scored[col].mean())
    per_sequence = scored.groupby("full_seq", sort=False)[list(cols)].mean()
    for col in cols:
        metrics[f"unique_sequence_uw_{col}"] = float(per_sequence[col].mean())
    sequence_weight = 1.0 / scored.groupby("full_seq")["full_seq"].transform("size").to_numpy(dtype=float)
    prefix_metrics = macro_recall_f1(scored["actual"], scored["predicted"])
    sequence_metrics = macro_recall_f1(
        scored["actual"], scored["predicted"], sequence_weight
    )
    metrics.update({f"prefix_example_{key}": value for key, value in prefix_metrics.items()})
    metrics.update({f"unique_sequence_{key}": value for key, value in sequence_metrics.items()})
    return metrics


def _mrr_from_probabilities(y, probabilities, groups, class_labels=None):
    """Mean per-sequence MRR, respecting a model's external class labels."""
    labels = np.arange(probabilities.shape[1]) if class_labels is None else np.asarray(class_labels)
    ranked = labels[np.argsort(-probabilities, axis=1, kind="stable")]
    reciprocal = np.zeros(len(y), dtype=float)
    for i, target in enumerate(y):
        matches = np.flatnonzero(ranked[i] == target)
        if len(matches):
            reciprocal[i] = 1.0 / (matches[0] + 1)
    frame = pd.DataFrame({"group": groups, "mrr": reciprocal})
    return float(frame.groupby("group")["mrr"].mean().mean())


def _cv_sequence_mrr(model, X, y, groups):
    """Mean per-sequence MRR for one GroupKFold validation fold."""
    probabilities = model.predict_proba(X)
    model_classes = np.asarray(model.classes_)
    ranked = model_classes[np.argsort(-probabilities, axis=1, kind="stable")]
    reciprocal = np.zeros(len(y), dtype=float)
    for i, target in enumerate(y):
        matches = np.flatnonzero(ranked[i] == target)
        if len(matches):
            reciprocal[i] = 1.0 / (matches[0] + 1)
    frame = pd.DataFrame({"group": groups, "mrr": reciprocal})
    return float(frame.groupby("group")["mrr"].mean().mean())


def _prepare():
    train, val, train_seq, val_seq = _load_examples()
    if train.empty or val.empty:
        raise ValueError("Train and validation examples must both be non-empty")
    encoder = LabelEncoder().fit(train["next"].astype(str))
    y = encoder.transform(train["next"].astype(str))
    techniques = sorted(set(train["next"].astype(str)) | {t for p in train["prefix"] for t in p})
    X = _make_features(train, techniques)
    X_val = _make_features(val, techniques)
    weights = _sequence_weights(train)
    groups = _groups(train)
    return train, val, train_seq, val_seq, encoder, y, X, X_val, weights, groups


def _optuna_search():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is optional; install it with `python -m pip install optuna`.") from exc

    train, val, train_seq, val_seq, encoder, y, X, X_val, weights, groups = _prepare()
    cv = GroupKFold(n_splits=3)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 9]),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        }
        fold_scores = []
        for fold_i, (outer_train, outer_val) in enumerate(cv.split(X, y, groups)):
            model, _ = _fit_with_group_early_stopping(
                X.iloc[outer_train], y[outer_train], weights[outer_train], groups[outer_train], params
            )
            score = _cv_sequence_mrr(
                model, X.iloc[outer_val], y[outer_val], groups[outer_val]
            )
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=12, show_progress_bar=False)

    model, best_iteration = _fit_with_group_early_stopping(X, y, weights, groups, study.best_params)
    metrics = _metrics(model, X_val, val, list(encoder.classes_))
    trials_path = config.LIGHTGBM_RESULTS_DIR / "lightgbm_optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    summary = {
        "search": "Optuna TPE, 12 trials, 3-fold GroupKFold on training sequences",
        "early_stopping": f"inner 15% group holdout, sequence-level MRR, patience {EARLY_STOPPING_ROUNDS}; refit all supplied training groups at best iteration",
        "cv_best_unique_sequence_mrr": study.best_value,
        "best_params": study.best_params,
        "best_iteration_from_inner_holdout": best_iteration,
        "external_validation_metrics": metrics,
        "training_unique_sequences": int(len(train_seq)),
        "validation_unique_sequences": int(len(val_seq)),
        "test_data_loaded": False,
    }
    (config.LIGHTGBM_RESULTS_DIR / "lightgbm_optuna_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Optuna best params:", study.best_params)
    print("Best training-CV MRR:", round(study.best_value, 4))
    print("External validation:", metrics)
    print("Saved trials:", trials_path)
    return summary


def _halving_search():
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV
    from scipy.stats import loguniform, randint, uniform

    train, val, train_seq, val_seq, encoder, y, X, X_val, weights, groups = _prepare()
    # Keep original row indices so the scorer can aggregate folds by sequence.
    group_by_index = dict(zip(X.index.tolist(), groups.tolist()))

    def scorer(estimator, X_fold, y_fold, sample_weight=None):
        # Primary scoring is unweighted by source duplicates; each sequence
        # contributes equally after its prefix rows are aggregated.
        fold_groups = np.asarray([group_by_index[i] for i in X_fold.index], dtype=int)
        return _cv_sequence_mrr(estimator, X_fold, np.asarray(y_fold), fold_groups)

    estimator = lgb.LGBMClassifier(**_base_params({}))
    space = {
        "num_leaves": randint(7, 64),
        "max_depth": [-1, 3, 5, 7, 9],
        "min_child_samples": randint(5, 41),
        "reg_alpha": loguniform(1e-4, 5.0),
        "reg_lambda": loguniform(1e-3, 20.0),
        "learning_rate": loguniform(0.02, 0.15),
        "colsample_bytree": uniform(0.6, 0.4),
        "subsample": uniform(0.7, 0.3),
    }
    search = HalvingRandomSearchCV(
        estimator, space, n_candidates=8, factor=2,
        resource="n_estimators", min_resources=40, max_resources=160,
        cv=GroupKFold(n_splits=3), scoring=scorer, refit=True,
        random_state=config.RANDOM_SEED, n_jobs=1, verbose=0,
        return_train_score=False,
    )
    search.fit(X, y, groups=groups, sample_weight=weights)
    # Halving uses estimator count as its resource, so early stopping cannot
    # run within those CV rounds. Apply sequence-grouped early stopping on the
    # final refit, then score the selected model on external validation.
    model, best_iteration = _fit_with_group_early_stopping(
        X, y, weights, groups, search.best_params_
    )
    metrics = _metrics(model, X_val, val, list(encoder.classes_))
    cv_path = config.LIGHTGBM_RESULTS_DIR / "lightgbm_halving_cv_results.csv"
    pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)
    summary = {
        "search": "HalvingRandomSearchCV, 8 random candidates, 3-fold GroupKFold on training sequences",
        "halving_resource": "n_estimators (40 to 160, factor 2)",
        "early_stopping": f"not used inside halving; final refit uses inner 15% group holdout, sequence-level MRR, patience {EARLY_STOPPING_ROUNDS}",
        "cv_best_unique_sequence_mrr": float(search.best_score_),
        "best_params": search.best_params_,
        "best_iteration_from_inner_holdout": best_iteration,
        "external_validation_metrics": metrics,
        "training_unique_sequences": int(len(train_seq)),
        "validation_unique_sequences": int(len(val_seq)),
        "test_data_loaded": False,
    }
    (config.LIGHTGBM_RESULTS_DIR / "lightgbm_halving_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Halving best params:", search.best_params_)
    print("Best training-CV MRR:", round(search.best_score_, 4))
    print("External validation:", metrics)
    print("Saved CV results:", cv_path)
    return summary


def _grid_search():
    train, val, train_seq, val_seq, encoder, y, X, X_val, weights, groups = _prepare()
    configs = [
        {"num_leaves": leaves, "min_child_samples": child, "reg_lambda": reg}
        for leaves, child, reg in product([7, 15, 31], [10, 20], [1.0, 5.0])
    ]
    records = []
    for i, params in enumerate(configs, 1):
        model, best_iteration = _fit_with_group_early_stopping(X, y, weights, groups, params)
        metrics = _metrics(model, X_val, val, list(encoder.classes_))
        records.append({"config_id": i, **params, "best_iteration": best_iteration, **metrics})
        print(f"Grid {i}/{len(configs)}: sequence-level MRR={metrics['unique_sequence_uw_mrr']:.4f}")
    results = pd.DataFrame(records)
    results.to_csv(config.LIGHTGBM_RESULTS_DIR / "lightgbm_validation_results.csv", index=False)
    best = results.sort_values("unique_sequence_uw_mrr", ascending=False).iloc[0].to_dict()
    summary = {
        "search": "12-setting fixed grid; models use early stopping on inner sequence holdout",
        "grid_parameters": {"num_leaves": [7, 15, 31], "min_child_samples": [10, 20], "reg_lambda": [1.0, 5.0]},
        "best_validation_config": best,
        "training_unique_sequences": int(len(train_seq)),
        "validation_unique_sequences": int(len(val_seq)),
        "test_data_loaded": False,
    }
    (config.LIGHTGBM_RESULTS_DIR / "lightgbm_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", choices=("grid", "optuna", "halving"), default="grid")
    args = parser.parse_args()
    if args.search == "grid":
        _grid_search()
    elif args.search == "optuna":
        _optuna_search()
    else:
        _halving_search()


if __name__ == "__main__":
    main()
