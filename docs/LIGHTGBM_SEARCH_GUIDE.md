# LightGBM Search Experiments

This experiment compares a feature-based boosted-tree model with the existing Markov and PrefixSpan predictors for next-technique ranking.

Tested here with LightGBM 4.7.0, scikit-learn 1.9.0, and optional Optuna 5.0.0.

## Run the experiments

From the repository root:

```bash
python src/models/lightgbm_predictor.py --search grid
python -m pip install optuna  # optional, only needed for the Optuna search
python src/models/lightgbm_predictor.py --search optuna
python src/models/lightgbm_predictor.py --search halving
```

The model uses prefix technique-presence indicators, indicators for each of the last four ordered techniques, and prefix length. Training rows are weighted so each source sequence contributes its `dup_count`. Cross-validation groups by full sequence, keeping prefixes from the same sequence together.

## Regularization and early stopping

- **L1 (`reg_alpha`)** and **L2 (`reg_lambda`)** regularize tree leaf weights. The fixed grid varied `reg_lambda` over 1 and 5; its `reg_alpha` stayed at LightGBM's default 0. The Optuna and halving searches tune both.
- **Tree complexity** is controlled by `num_leaves`, `max_depth`, and `min_child_samples`.
- **Early stopping** monitors mean sequence-level MRR on an inner 15% sequence-grouped holdout drawn from the training portion, with patience 30 and a 400-tree ceiling. The selected tree count is then used to refit on all rows in that training portion.
- **Optuna** runs 12 TPE trials with 3-fold `GroupKFold` over training sequences. It uses early stopping inside each fold and prunes weak trials between folds.
- **HalvingRandomSearchCV** samples 8 settings and uses 3-fold `GroupKFold`. It successively allocates 40, 80, then 160 trees. Early stopping cannot run inside this search because tree count is the resource being increased; it is applied during the final inner-holdout refit.

## Current results

All measurements below are from the validation experiments run on 2026-10-20. The separate held-out test data was not loaded.

| Search | Best training-CV sequence MRR | Validation unique-sequence MRR | Validation unique-sequence top-1 / top-3 / top-5 | Validation prefix-example MRR | Selected iterations |
|---|---:|---:|---:|---:|---:|
| 12-setting grid | — | 0.5407 | 0.3693 / 0.6369 / 0.7631 | 0.5583 | 101 |
| Optuna, 12 trials | 0.5110 | 0.5375 | 0.3655 / 0.6364 / 0.7630 | 0.5539 | 149 |
| Halving random search, 8 candidates | 0.5081 | 0.5362 | 0.3600 / 0.6423 / 0.7627 | 0.5573 | 90 |

The grid selected settings directly against the external validation MRR, so its validation score is a tuning score and may be optimistic. Optuna and halving select configurations using grouped cross-validation within training data; the external validation split remains separate for those searches. These validation results are not final test results, and the two search algorithms do not establish that one method is statistically better.

The unweighted **unique-sequence** metrics are primary: average prefix scores within each full sequence, then average across sequences. Prefix-example metrics average all prefix rows directly and are included only to compare with earlier project tables.

## Output files

- `results/lightgbm/lightgbm_validation_results.csv` and `results/lightgbm/lightgbm_validation_summary.json`: fixed grid.
- `results/lightgbm/lightgbm_optuna_trials.csv` and `results/lightgbm/lightgbm_optuna_summary.json`: Optuna search history and selected result.
- `results/lightgbm/lightgbm_halving_cv_results.csv` and `results/lightgbm/lightgbm_halving_summary.json`: successive-halving search history and selected result.

`HalvingRandomSearchCV` is an experimental scikit-learn API and may change. Optuna is an optional dependency; install it only to run that search mode.


## Additional class metrics (2026-10-20)

The refreshed grid and halving outputs include macro recall and macro F1 for both prefix-example and sequence-balanced weighting. Macro averaging covers only target labels represented in the validation targets; sequence-balanced values weight each prefix by the inverse number of prefixes in its full sequence. The refreshed Optuna run was unavailable because the optional package is not installed, so its existing summary has not been updated with these metrics. The consolidated comparison is `results/comparison/all_models_validation_sequence_metrics.csv`.
