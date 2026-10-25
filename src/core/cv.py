"""Grouped, length-stratified cross-validation over the development set.

Why this exists
---------------
Model selection and comparative reporting previously used a single
545-sequence validation holdout. Measured on this data, per-sequence MRR has
sd 0.2101, so that holdout carries SE 0.0090 and a 95% CI half-width of
0.0176: the smallest unpaired difference it can resolve at 80% power is about
0.036 MRR. The three LightGBM searches differed by 0.0047, which is 7.5x below
that threshold, so their ranking was not measurable.

Pooling the original train and validation splits gives a development set of
2,178 unique sequences and halves the standard error to 0.0045. Folds are
grouped by full sequence, so no sequence contributes prefixes to both sides of
a fold, and stratified by the same length buckets the original split used, so
fold composition stays comparable.

The test split is never loaded here. `assert_test_sealed` is available for
scripts that want to state that invariant explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config

PROCESSED_DIR = config.BASE_DIR / "data" / "processed"


def length_bucket(n: int) -> str:
    """Stratification bucket, matching the buckets used by the 60/20/20 split."""
    return str(n) if n <= 6 else "7+"


def load_development_examples() -> pd.DataFrame:
    """Prefix examples from the development splits, with sequence id and bucket.

    Adds `seq_id` (a stable integer per full sequence, so folds can group on it),
    `source_split` (which of the original splits the row came from, kept so older
    results stay traceable), and `length_bucket` for stratification.
    """
    frames = []
    for split in config.DEV_SPLITS:
        frame = pd.read_pickle(PROCESSED_DIR / f"{split}_examples.pkl")
        frame = frame.copy()
        frame["source_split"] = split
        frames.append(frame)
    examples = pd.concat(frames, ignore_index=True)
    full_seq = examples["full_seq"].map(tuple)
    examples["seq_id"] = pd.factorize(full_seq, sort=True)[0]
    examples["length_bucket"] = full_seq.map(lambda s: length_bucket(len(s)))
    return examples


def development_folds(examples: pd.DataFrame, n_splits: int | None = None):
    """Yield (fold_index, fit_row_idx, score_row_idx) for grouped CV.

    Grouping is by `seq_id`, so every prefix of a sequence lands on one side.
    Stratification is by `length_bucket`; each sequence has exactly one bucket,
    so the two constraints do not conflict.
    """
    n_splits = config.N_CV_FOLDS if n_splits is None else n_splits
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED
    )
    groups = examples["seq_id"].to_numpy()
    strata = examples["length_bucket"].to_numpy()
    for fold, (fit_idx, score_idx) in enumerate(
        splitter.split(examples, strata, groups)
    ):
        yield fold, fit_idx, score_idx


def fold_assignments(examples: pd.DataFrame, n_splits: int | None = None) -> np.ndarray:
    """Per-row fold label, for saving alongside results or for sanity checks."""
    assignment = np.full(len(examples), -1, dtype=int)
    for fold, _, score_idx in development_folds(examples, n_splits):
        assignment[score_idx] = fold
    if (assignment < 0).any():
        raise ValueError("Every example must be scored in exactly one fold")
    return assignment


def verify_folds(examples: pd.DataFrame, n_splits: int | None = None) -> pd.DataFrame:
    """Check the invariants that make these folds usable, and summarise them.

    Raises if any sequence straddles a fold boundary; returns a per-fold table of
    sizes and length-bucket shares so stratification can be inspected.
    """
    n_splits = config.N_CV_FOLDS if n_splits is None else n_splits
    rows = []
    seen_score_rows = set()
    for fold, fit_idx, score_idx in development_folds(examples, n_splits):
        fit_groups = set(examples["seq_id"].to_numpy()[fit_idx])
        score_groups = set(examples["seq_id"].to_numpy()[score_idx])
        overlap = fit_groups & score_groups
        if overlap:
            raise ValueError(
                f"Fold {fold}: {len(overlap)} sequences appear on both sides"
            )
        if seen_score_rows & set(score_idx.tolist()):
            raise ValueError(f"Fold {fold}: scoring rows overlap another fold")
        seen_score_rows |= set(score_idx.tolist())
        held = examples.iloc[score_idx]
        shares = held["length_bucket"].value_counts(normalize=True)
        rows.append({
            "fold": fold,
            "fit_examples": len(fit_idx),
            "score_examples": len(score_idx),
            "fit_sequences": len(fit_groups),
            "score_sequences": len(score_groups),
            **{f"bucket_{b}": round(float(shares.get(b, 0.0)), 4)
               for b in ("4", "5", "6", "7+")},
        })
    if len(seen_score_rows) != len(examples):
        raise ValueError("Folds must partition every development example exactly once")
    return pd.DataFrame(rows)


def assert_test_sealed() -> None:
    """State the invariant that the test split is not in play.

    The development loader only ever reads the splits named in
    `config.DEV_SPLITS`; this guard makes a script's intent explicit and fails
    loudly if that list is ever widened to include a sealed split.
    """
    leaked = set(config.DEV_SPLITS) & set(config.SEALED_SPLITS)
    if leaked:
        raise RuntimeError(f"Sealed splits must not be used for development: {leaked}")


if __name__ == "__main__":
    assert_test_sealed()
    examples = load_development_examples()
    summary = verify_folds(examples)
    unique_sequences = examples["seq_id"].nunique()
    print(f"Development examples: {len(examples):,} from {unique_sequences:,} unique sequences")
    print(f"Source splits: {config.DEV_SPLITS}  |  sealed: {config.SEALED_SPLITS}")
    print(f"\n{config.N_CV_FOLDS}-fold StratifiedGroupKFold (grouped by sequence, stratified by length):")
    print(summary.to_string(index=False))
    out = config.CV_RESULTS_DIR / "development_fold_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nAll invariants passed: no sequence straddles a fold, folds partition every example.")
    print(f"Saved: {out}")
