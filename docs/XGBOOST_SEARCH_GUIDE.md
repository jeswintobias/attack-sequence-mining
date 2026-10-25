# XGBoost Optuna Search

This experiment adds an XGBoost multiclass predictor for next reported ATT&CK technique ranking. It uses the same prefix-presence, last-four-position, and prefix-length features as LightGBM.

## Run

```bash
python -m pip install xgboost optuna
python src/models/xgboost_predictor.py --search optuna
python src/eval/compare_validation_models.py
```

## Evaluation protocol

- Optuna TPE searches twelve parameter trials using three-fold `GroupKFold` over training sequences, with median pruning after intermediate folds. Full sequences remain grouped, so their prefixes do not cross fold boundaries.
- Each fold uses a separate 15% group holdout for early stopping on **sequence-level MRR**, with patience 30 and a 400-tree ceiling.
- The search space offers both `depthwise` and `lossguide` growth. Under `lossguide` the depth cap is released and `max_leaves` governs tree size, so XGBoost can build the wide, leaf-wise trees LightGBM uses.
- The Optuna objective is mean sequence-level MRR across training folds. The selected configuration is refit using training data only; external validation is used for the report.
- Candidates already present in the prefix are ruled out before ranking (DEC-17), in both the stopping metric and the reported metrics.
- Macro recall and macro F1 are reported over labels observed in validation. The sequence-balanced versions weight each prefix by the inverse number of prefixes in its full sequence. Top-1, top-3, top-5, and MRR use the same sequence-balanced averaging.
- The held-out test split is never loaded.

### Protocol parity with LightGBM

Fold count (3), trial budget (12), tree ceiling (400), patience (30), and the early-stopping metric (sequence MRR) now match the LightGBM Optuna search, so a difference between the two models is attributable to the models rather than to their harnesses.

## Corrected run (2026-10-25)

The first run (2026-10-20) selected tree count by **multiclass log loss** under a **160-tree ceiling**. Both choices were wrong for this task, and measurably so:

- Log loss is non-monotone on this 94-class problem. On the inner holdout it dipped at round 114, which fired early stopping at round 135 and reported 115 trees, while its own global minimum was at round **196** and sequence MRR was still rising at round 400. Early stopping triggered on noise, under a ceiling that would have truncated it regardless.
- Measured divergence of the two criteria on the same holdout: log loss 2.2733 / 2.2781 / 2.2556 / 2.2699 at rounds 115 / 134 / 196 / 400, against sequence MRR 0.4736 / 0.4741 / 0.4856 / 0.4923 over the same rounds.
- Because the same stopping rule ran inside every CV fold, the search itself ranked configurations by truncated models, so the originally selected hyperparameters were not reliably the best either.

| Measure | 2026-10-20 run | Corrected run | Change |
|---|---:|---:|---:|
| Best training CV sequence MRR | 0.4656 | 0.4830 | +0.0174 |
| Validation sequence top-1 | 0.3356 | 0.3547 | +0.0191 |
| Validation sequence top-3 | 0.5822 | 0.6051 | +0.0229 |
| Validation sequence top-5 | 0.7248 | 0.7456 | +0.0208 |
| Validation sequence MRR | 0.5025 | **0.5233** | +0.0208 |
| Validation sequence macro recall | 0.1505 | 0.1963 | +0.0458 |
| Validation sequence macro F1 | 0.1446 | 0.1880 | +0.0434 |
| Selected boosting iterations | 115 | 109 | — |
| Test data loaded | No | No | — |

Selected configuration: `grow_policy=depthwise, max_depth=4, learning_rate=0.1866, min_child_weight=0.534, subsample=0.933, colsample_bytree=0.606, gamma=0.909, reg_alpha=1.26e-4, reg_lambda=0.0230`. The search had `lossguide` available and did not choose it; the gain came from the stopping metric, which let a markedly higher learning rate (0.187 vs 0.054) be evaluated on its ranking quality rather than on its calibration.

The earlier conclusion that XGBoost trails LightGBM by roughly 0.038 MRR was therefore a harness artifact. The remaining gap is about 0.018, inside the uncertainty band of this validation split: the one paired bootstrap run in this project had a 95% CI half-width of roughly 0.016 over the same 545 sequences, so differences of this size are not resolvable without a paired test on each pair.

### A note on `n_jobs`

`n_jobs=-1` was adopted here, but it is **not** a speedup at this data size. Measured on this dataset, LightGBM runs 30 boosting rounds in 1.5s at `n_jobs=1` versus 8.3s at `n_jobs=-1` — thread overhead dominates on 8,329 rows — with bit-identical predictions. Thread count is a performance setting only and changes no result; do not expect extra search budget from it.

The Optuna budget remains modest and matched to LightGBM, so this is a controlled model comparison, not an exhaustive search. Results do not establish chronological prediction of real attacker actions because the source sequences are reported technique lists.

## Outputs

- `results/xgboost/xgboost_optuna_trials.csv`: trial parameters, outcomes, and pruning status.
- `results/xgboost/xgboost_optuna_summary.json`: best parameters, CV score, early-stopping count, validation metrics, and test-isolation flag.
- `results/comparison/all_models_validation_sequence_metrics.csv`: refreshed common comparison table.
