# XGBoost Optuna Search

This experiment adds an XGBoost multiclass predictor for next reported ATT&CK technique ranking. It uses the same prefix-presence, last-four-position, and prefix-length features as LightGBM.

## Run

```bash
python -m pip install xgboost optuna
python src/models/xgboost_predictor.py --search optuna
python src/eval/compare_validation_models.py
```

## Evaluation protocol

- Optuna TPE searches eight parameter trials using three-fold `GroupKFold` over training sequences. Four weak trials may be pruned after intermediate folds.
- Each fold uses a separate 15% group holdout for early stopping on validation multiclass log loss, with patience 20 and a 160-tree ceiling. Full sequences remain grouped, so their prefixes do not cross fold boundaries.
- The Optuna objective is mean sequence-level MRR across training folds. The selected configuration is refit using training data only; external validation is used for the report.
- Macro recall and macro F1 are reported over labels observed in validation. The sequence-balanced versions weight each prefix by the inverse number of prefixes in its full sequence. Top-1, top-3, top-5, and MRR use the same sequence-balanced averaging.
- The held-out test split is never loaded.

## Measured run (2026-10-20)

| Measure | Value |
|---|---:|
| Best training CV sequence MRR | 0.4656 |
| External validation sequence top-1 / top-3 / top-5 | 0.3356 / 0.5822 / 0.7248 |
| External validation sequence MRR | 0.5025 |
| External validation sequence macro recall / macro F1 | 0.1505 / 0.1446 |
| Selected boosting iterations | 115 |
| Test data loaded | No |

The Optuna best CV score is a tuning result, while external validation provides the comparison result. The search budget is intentionally modest for this multiclass dataset; this is an initial model comparison, not an exhaustive search. Results do not establish chronological prediction of real attacker actions because the source sequences are reported technique lists.

## Outputs

- `results/xgboost/xgboost_optuna_trials.csv`: trial parameters, outcomes, and pruning status.
- `results/xgboost/xgboost_optuna_summary.json`: best parameters, CV score, early-stopping count, validation metrics, and test-isolation flag.
- `results/comparison/all_models_validation_sequence_metrics.csv`: refreshed common comparison table.
