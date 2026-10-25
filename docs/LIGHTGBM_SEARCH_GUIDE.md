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
- **Early stopping** monitors mean sequence-level MRR on an inner 15% sequence-grouped holdout drawn from the training portion, with patience 30 and a 400-tree ceiling. The selected tree count is then used to refit on all rows in that training portion. The holdout score uses the same prefix candidate masking as the reported metrics, so tuning and reporting optimise the same quantity.
- **Optuna** runs 12 TPE trials with 3-fold `GroupKFold` over training sequences. It uses early stopping inside each fold and prunes weak trials between folds.
- **HalvingRandomSearchCV** samples 8 settings and uses 3-fold `GroupKFold`. It successively allocates 40, 80, then 160 trees. Early stopping cannot run inside this search because tree count is the resource being increased; it is applied during the final inner-holdout refit.

## Current results

All measurements below are from the validation experiments re-run on 2026-10-25 under prefix candidate
masking (DEC-17). The separate held-out test data was not loaded.

| Search | Best training-CV sequence MRR | Validation sequence MRR | Validation sequence top-1 / top-3 / top-5 | Validation prefix-example MRR | Validation sequence macro recall | Selected iterations |
|---|---:|---:|---:|---:|---:|---:|
| 12-setting grid | — | *0.5410* | 0.3693 / 0.6369 / 0.7651 | 0.5586 | 0.2052 | 101 |
| Optuna, 12 trials | 0.5095 | **0.5379** | 0.3659 / 0.6363 / 0.7643 | 0.5549 | 0.1952 | 152 |
| Halving random search, 8 candidates | 0.5084 | 0.5363 | 0.3600 / 0.6427 / 0.7627 | 0.5575 | 0.2015 | 90 |

The grid selected settings directly against the external validation MRR, so its score is a tuning
score and is optimistic — it is shown in italics rather than reported as the best model. Optuna and
halving select using grouped cross-validation inside the training data, leaving the external
validation split genuinely held out for those two; those are the comparable numbers.

The three searches span 0.5363-0.5410 sequence MRR, a range of 0.005. The one paired bootstrap run in
this project had a 95% CI half-width of roughly 0.016 over the same 545 validation sequences, so this
spread is noise and **these results do not establish that any one search method is better than
another**. They are also not final test results.

The unweighted **unique-sequence** metrics are primary: average prefix scores within each full
sequence, then average across sequences. Prefix-example metrics average all prefix rows directly and
are included only to compare with earlier project tables.

### Effect of prefix candidate masking

Masking candidates already present in the prefix (DEC-17) moved the grid best from 0.540695 to
0.540960 sequence MRR — a change of +0.0003. The boosted trees receive `seen_*` presence indicators
and had therefore already learned the constraint from data. The same change is worth +0.0271 to
Markov-2 and +0.0277 to PrefixSpan, which had no way to express it. Part of the trees' measured
advantage over those baselines was this artifact rather than better modelling.

### A note on `n_jobs`

`n_jobs=1` is kept deliberately and is **not** a limitation. Measured on this dataset, LightGBM runs 30
boosting rounds in 1.5s at `n_jobs=1` versus 8.3s at `n_jobs=-1` — thread overhead dominates at 8,329
rows — with bit-identical predictions. Thread count changes runtime only, never a result.

### Comparison against XGBoost

The XGBoost Optuna search now matches this one on fold count, trial budget, tree ceiling, patience and
early-stopping metric (see `docs/XGBOOST_SEARCH_GUIDE.md`). On that footing LightGBM_Optuna leads
XGBoost_Optuna by 0.0146 sequence MRR (0.5379 vs 0.5233), with macro recall effectively tied (0.1952
vs 0.1963). That margin is within the uncertainty of this split, so the two models are not separated
by this evidence.

## Output files

- `results/lightgbm/lightgbm_validation_results.csv` and `results/lightgbm/lightgbm_validation_summary.json`: fixed grid.
- `results/lightgbm/lightgbm_optuna_trials.csv` and `results/lightgbm/lightgbm_optuna_summary.json`: Optuna search history and selected result.
- `results/lightgbm/lightgbm_halving_cv_results.csv` and `results/lightgbm/lightgbm_halving_summary.json`: successive-halving search history and selected result.

`HalvingRandomSearchCV` is an experimental scikit-learn API and may change. Optuna is an optional dependency; install it only to run that search mode.


## Additional class metrics

All three searches report macro recall and macro F1 for both prefix-example and sequence-balanced weighting. Macro averaging covers only target labels represented in the validation targets; sequence-balanced values weight each prefix by the inverse number of prefixes in its full sequence.

Optuna was installed on 2026-10-25, so its run is no longer skipped: it now carries the class metrics and appears in the consolidated comparison, `results/comparison/all_models_validation_sequence_metrics.csv`, alongside the grid, halving and XGBoost rows. That table lists every search protocol rather than only the best-scoring ones, and records which split each configuration was selected on.
