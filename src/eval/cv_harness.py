"""Cross-validated scoring and paired model comparison on the development set.

Every model is driven through one interface so results are directly comparable:
an adapter exposes `fit(fit_examples)` and `rank(score_examples)`, where `rank`
returns a ranked list of technique IDs for each row. Candidates already present
in a prefix are removed centrally (config.MASK_SEEN_TECHNIQUES), so no adapter
can forget to do it and no adapter can do it differently from another.

Comparisons are **paired**: both models are scored on the same folds and the
same sequences, and the difference is taken per sequence before aggregating.
On this data an unpaired comparison on 545 sequences cannot resolve anything
below about 0.036 MRR, which is far coarser than the differences being chased,
so pairing is what makes the comparison informative rather than decorative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"), str(SRC / "mining")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from cv import development_folds, load_development_examples
from baselines import mask_seen
from evaluation_metrics import macro_recall_f1

TOP_K = (1, 3, 5)


def score_rankings(rankings, examples: pd.DataFrame) -> pd.DataFrame:
    """Per-row reciprocal rank and hit@k from ranked technique lists."""
    if len(rankings) != len(examples):
        raise ValueError(f"Got {len(rankings)} rankings for {len(examples)} examples")
    records = []
    prefixes = examples["prefix"].tolist()
    targets = examples["next"].astype(str).tolist()
    seq_ids = examples["seq_id"].tolist()
    for ranked, prefix, target, seq_id in zip(rankings, prefixes, targets, seq_ids):
        ranked = mask_seen(list(ranked), prefix)
        position = ranked.index(target) + 1 if target in ranked else None
        records.append({
            "seq_id": seq_id,
            "actual": target,
            "predicted": ranked[0] if ranked else None,
            "rr": 1.0 / position if position else 0.0,
            **{f"top_{k}": float(bool(position and position <= k)) for k in TOP_K},
        })
    return pd.DataFrame(records)


def per_sequence(scored: pd.DataFrame) -> pd.DataFrame:
    """Average prefix scores within each full sequence, so long sequences do not
    dominate by generating more prefixes."""
    cols = ["rr"] + [f"top_{k}" for k in TOP_K]
    return scored.groupby("seq_id", sort=True)[cols].mean()


def aggregate(scored: pd.DataFrame) -> dict:
    """Sequence-balanced headline metrics plus macro class metrics."""
    seq = per_sequence(scored)
    weight = 1.0 / scored.groupby("seq_id")["seq_id"].transform("size").to_numpy(float)
    out = {"seq_mrr": float(seq["rr"].mean())}
    out.update({f"seq_top_{k}": float(seq[f"top_{k}"].mean()) for k in TOP_K})
    out.update({
        f"seq_{key}": value for key, value in
        macro_recall_f1(scored["actual"], scored["predicted"], weight).items()
    })
    return out


def run_cv(adapter_factory, name: str, examples: pd.DataFrame | None = None,
           n_splits: int | None = None, verbose: bool = True,
           max_folds: int | None = None):
    """Fit and score a model across the development folds.

    `adapter_factory` is called once per fold with no arguments and must return a
    fresh adapter, so nothing leaks between folds.

    `max_folds` stops after that many folds, for a quick screen. The fold
    geometry is unchanged -- it is still the same 5-way partition, just not all
    of it scored -- so a screen and a full run are directly comparable fold by
    fold.

    Returns (fold_table, per_sequence_scores, pooled_metrics). The per-sequence
    frame is indexed by `seq_id` across all folds, which is what `paired_compare`
    needs; when every fold is run they partition the development set, so each
    sequence appears exactly once.
    """
    examples = load_development_examples() if examples is None else examples
    fold_rows, pieces = [], []
    for fold, fit_idx, score_idx in development_folds(examples, n_splits):
        if max_folds is not None and fold >= max_folds:
            break
        fit_examples = examples.iloc[fit_idx].reset_index(drop=True)
        score_examples = examples.iloc[score_idx].reset_index(drop=True)
        adapter = adapter_factory()
        adapter.fit(fit_examples)
        scored = score_rankings(adapter.rank(score_examples), score_examples)
        metrics = aggregate(scored)
        fold_rows.append({"model": name, "fold": fold,
                          "score_sequences": int(scored["seq_id"].nunique()), **metrics})
        pieces.append(per_sequence(scored))
        if verbose:
            print(f"  {name} fold {fold}: seq_mrr={metrics['seq_mrr']:.4f} "
                  f"top1={metrics['seq_top_1']:.4f} top5={metrics['seq_top_5']:.4f}", flush=True)
    folds = pd.DataFrame(fold_rows)
    seq_scores = pd.concat(pieces).sort_index()
    pooled = {
        "model": name,
        **{c: float(folds[c].mean()) for c in folds.columns if c.startswith("seq_")},
        **{f"{c}_sd": float(folds[c].std(ddof=1)) for c in ("seq_mrr", "seq_top_1")},
        "n_folds": int(len(folds)),
        "n_sequences": int(seq_scores.shape[0]),
    }
    return folds, seq_scores, pooled


def paired_compare(seq_a: pd.DataFrame, seq_b: pd.DataFrame, name_a: str, name_b: str,
                   metric: str = "rr", n_resamples: int = 2000,
                   seed: int | None = None) -> dict:
    """Paired difference A - B over the same sequences, with a bootstrap CI.

    Resampling is over sequences, which are the independent units here; prefixes
    within a sequence are not independent and must not be resampled separately.
    """
    shared = seq_a.index.intersection(seq_b.index)
    if len(shared) == 0:
        raise ValueError("Models share no sequences; cannot pair")
    a = seq_a.loc[shared, metric].to_numpy(float)
    b = seq_b.loc[shared, metric].to_numpy(float)
    diff = a - b
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    draws = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "model_a": name_a,
        "model_b": name_b,
        "metric": metric,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci_2.5": float(low),
        "ci_97.5": float(high),
        "significant": bool(low > 0 or high < 0),
        "n_sequences": int(len(diff)),
        "n_resamples": int(n_resamples),
    }


def comparison_table(seq_scores: dict, metric: str = "rr") -> pd.DataFrame:
    """Every pairwise paired comparison among the supplied models."""
    names = list(seq_scores)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append(paired_compare(seq_scores[a], seq_scores[b], a, b, metric))
    return pd.DataFrame(rows)
