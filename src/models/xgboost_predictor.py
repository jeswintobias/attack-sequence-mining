"""XGBoost next-technique prediction with sequence-grouped Optuna tuning.

Run from the repository root:
  python src/xgboost_predictor.py --search optuna

Only training and validation splits are loaded. Full sequences are kept intact
inside GroupKFold; a separate inner group holdout selects the boosting round.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
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
from lightgbm_predictor import _groups, _load_examples, _make_features, _sequence_weights

MAX_ESTIMATORS = 160
EARLY_STOPPING_ROUNDS = 20
N_TRIALS = 8
N_FOLDS = 3


def _base_params(
    params: dict,
    n_estimators: int = MAX_ESTIMATORS,
    num_class: int = 2,
    early_stopping_rounds: int | None = None,
):
    base = {
        "objective": "multi:softprob",
        "num_class": int(num_class),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": int(n_estimators),
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": config.RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": 0,
        **params,
    }
    if early_stopping_rounds is not None:
        base["early_stopping_rounds"] = int(early_stopping_rounds)
    return base


def _local_encode(y, global_label_ids):
    mapping = {int(label): i for i, label in enumerate(global_label_ids)}
    return np.asarray([mapping[int(label)] for label in y], dtype=np.int32), mapping


def _fit_with_group_early_stopping(X, y, weights, groups, params, refit=True):
    """Select tree count on inner grouped holdout, then refit supplied groups."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.RANDOM_SEED)
    fit_idx, stop_idx = next(split.split(X, y, groups))
    fit_global_labels = np.unique(y[fit_idx])
    seen = np.isin(y[stop_idx], fit_global_labels)
    stop_idx = stop_idx[seen]
    if len(fit_global_labels) < 2 or not len(stop_idx):
        raise ValueError("Early-stopping holdout needs at least two seen target classes")

    y_fit, _ = _local_encode(y[fit_idx], fit_global_labels)
    stop_global_labels = y[stop_idx]
    y_stop, _ = _local_encode(stop_global_labels, fit_global_labels)
    model = xgb.XGBClassifier(**_base_params(
        params,
        num_class=len(fit_global_labels),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    ))
    model.fit(
        X.iloc[fit_idx], y_fit,
        sample_weight=weights[fit_idx],
        eval_set=[(X.iloc[stop_idx], y_stop)],
        sample_weight_eval_set=[weights[stop_idx]],
        verbose=False,
    )
    best_iteration = int(getattr(model, "best_iteration", MAX_ESTIMATORS - 1)) + 1

    if not refit:
        # The inner early-stopping model can score the untouched outer fold;
        # skip an expensive extra refit during each CV fold.
        return model, fit_global_labels, best_iteration

    all_global_labels = np.unique(y)
    y_all, _ = _local_encode(y, all_global_labels)
    final_model = xgb.XGBClassifier(
        **_base_params(params, n_estimators=best_iteration, num_class=len(all_global_labels))
    )
    final_model.fit(X, y_all, sample_weight=weights, verbose=False)
    return final_model, all_global_labels, best_iteration


def _sequence_mrr(model, class_ids, X, y, groups):
    probabilities = model.predict_proba(X)
    ranked_global = np.asarray(class_ids)[np.argsort(-probabilities, axis=1, kind="stable")]
    reciprocal = np.zeros(len(y), dtype=float)
    for i, target in enumerate(y):
        hit = np.flatnonzero(ranked_global[i] == target)
        if len(hit):
            reciprocal[i] = 1.0 / (int(hit[0]) + 1)
    return float(pd.DataFrame({"group": groups, "mrr": reciprocal}).groupby("group")["mrr"].mean().mean())


def _metrics(model, class_ids, X, examples, label_names):
    probabilities = model.predict_proba(X)
    output_labels = np.asarray(label_names, dtype=object)[np.asarray(class_ids, dtype=int)]
    records = []
    for i, (_, row) in enumerate(examples.iterrows()):
        order = np.argsort(-probabilities[i], kind="stable")
        ranked = output_labels[order]
        score = evaluate_predictions(ranked.tolist(), row["next"])
        records.append({
            **score,
            "actual": row["next"],
            "predicted": ranked[0] if len(ranked) else None,
            "full_seq": tuple(row["full_seq"]),
        })
    scored = pd.DataFrame(records)
    cols = ("top_1", "top_3", "top_5", "mrr")
    result = {}
    for col in cols:
        result[f"prefix_example_uw_{col}"] = float(scored[col].mean())
        result[f"unique_sequence_uw_{col}"] = float(
            scored.groupby("full_seq", sort=False)[col].mean().mean()
        )
    seq_weight = 1.0 / scored.groupby("full_seq")["full_seq"].transform("size").to_numpy(dtype=float)
    result.update({f"prefix_example_{k}": v for k, v in macro_recall_f1(
        scored["actual"], scored["predicted"]
    ).items()})
    result.update({f"unique_sequence_{k}": v for k, v in macro_recall_f1(
        scored["actual"], scored["predicted"], seq_weight
    ).items()})
    return result


def _prepare():
    train, val, train_seq, val_seq = _load_examples()
    encoder = LabelEncoder().fit(train["next"].astype(str))
    y = encoder.transform(train["next"].astype(str))
    techniques = sorted(set(train["next"].astype(str)) | {t for p in train["prefix"] for t in p})
    X = _make_features(train, techniques)
    X_val = _make_features(val, techniques)
    return (
        train, val, train_seq, val_seq, encoder, y, X, X_val,
        _sequence_weights(train), _groups(train), _groups(val),
    )


def run_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for this experiment. Install with `python -m pip install optuna`."
        ) from exc

    train, val, train_seq, val_seq, encoder, y, X, X_val, weights, groups, val_groups = _prepare()
    cv = GroupKFold(n_splits=N_FOLDS)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 12.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
        fold_scores = []
        for fold, (fit_idx, score_idx) in enumerate(cv.split(X, y, groups)):
            model, class_ids, _ = _fit_with_group_early_stopping(
                X.iloc[fit_idx], y[fit_idx], weights[fit_idx], groups[fit_idx], params,
                refit=False,
            )
            fold_scores.append(_sequence_mrr(
                model, class_ids, X.iloc[score_idx], y[score_idx], groups[score_idx]
            ))
            print(
                f"Trial {trial.number + 1}/{N_TRIALS}, fold {fold + 1}/{N_FOLDS}: "
                f"sequence MRR={fold_scores[-1]:.4f}",
                flush=True,
            )
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    model, class_ids, best_iteration = _fit_with_group_early_stopping(
        X, y, weights, groups, study.best_params
    )
    validation_metrics = _metrics(model, class_ids, X_val, val, encoder.classes_)
    study.trials_dataframe().to_csv(config.XGBOOST_RESULTS_DIR / "xgboost_optuna_trials.csv", index=False)
    summary = {
        "search": f"Optuna TPE, {N_TRIALS} trials, {N_FOLDS}-fold GroupKFold on training sequences",
        "objective": "mean unique-sequence MRR across training CV folds",
        "early_stopping": f"inner 15% sequence-group holdout, validation log loss, patience {EARLY_STOPPING_ROUNDS}, ceiling {MAX_ESTIMATORS}; use the stopping model during CV, then refit all training groups at selected tree count for final validation",
        "best_training_cv_unique_sequence_mrr": float(study.best_value),
        "best_params": study.best_params,
        "best_iteration_from_inner_holdout": best_iteration,
        "external_validation_metrics": validation_metrics,
        "training_unique_sequences": int(len(train_seq)),
        "validation_unique_sequences": int(len(val_seq)),
        "test_data_loaded": False,
        "macro_metric_definition": "macro recall/F1 over labels observed in validation; sequence-balanced by inverse prefix count",
    }
    (config.XGBOOST_RESULTS_DIR / "xgboost_optuna_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Best training CV unique-sequence MRR:", round(study.best_value, 4))
    print("Best parameters:", study.best_params)
    print("Best iteration:", best_iteration)
    print("External validation metrics:", validation_metrics)
    print("Test data loaded: False")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", choices=("optuna",), default="optuna")
    parser.parse_args()
    run_optuna()


if __name__ == "__main__":
    main()
