"""
Part 2 – EDA fixes: length stats, majority baseline recalc, chart annotations.
Run this script to generate results/length_stats_ge4.csv.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from split import collapse_technique, deduplicate_first_occurrence


def compute_length_stats():
    """Compute length stats for rows and unique sequences (length >= 4, collapsed)."""
    df = pd.read_csv(config.SIGHTINGS_DATA_PATH)

    # Parse and collapse
    sequences = []
    for raw in df["tid"].apply(ast.literal_eval):
        techs = [collapse_technique(t) for t in raw]
        cleaned = deduplicate_first_occurrence(techs)
        if len(cleaned) >= 4:
            sequences.append(tuple(cleaned))

    # ── Row-level stats (all extracted rows, including duplicates) ─────────
    row_lengths = pd.Series([len(s) for s in sequences])
    row_stats = {
        "scope": "all_rows_ge4",
        "n": len(row_lengths),
        "mean": round(row_lengths.mean(), 4),
        "median": row_lengths.median(),
        "std": round(row_lengths.std(), 4),
        "skew": round(row_lengths.skew(), 4),
        "min": int(row_lengths.min()),
        "q25": row_lengths.quantile(0.25),
        "q50": row_lengths.quantile(0.50),
        "q75": row_lengths.quantile(0.75),
        "max": int(row_lengths.max()),
    }

    # ── Unique-sequence stats ──────────────────────────────────────────────
    unique_seqs = list(set(sequences))
    uniq_lengths = pd.Series([len(s) for s in unique_seqs])
    uniq_stats = {
        "scope": "unique_seqs_ge4",
        "n": len(uniq_lengths),
        "mean": round(uniq_lengths.mean(), 4),
        "median": uniq_lengths.median(),
        "std": round(uniq_lengths.std(), 4),
        "skew": round(uniq_lengths.skew(), 4),
        "min": int(uniq_lengths.min()),
        "q25": uniq_lengths.quantile(0.25),
        "q50": uniq_lengths.quantile(0.50),
        "q75": uniq_lengths.quantile(0.75),
        "max": int(uniq_lengths.max()),
    }

    df_stats = pd.DataFrame([row_stats, uniq_stats])
    path = config.RESULTS_DIR / "length_stats_ge4.csv"
    df_stats.to_csv(path, index=False)
    print(f"Saved {path}")
    print(df_stats.to_string(index=False))

    # Chart 2 check
    print(f"\nChart 2 check (collapsed, first-occurrence dedup, length >= 4):")
    print(f"  All rows:         mean={row_stats['mean']}, median={row_stats['median']}")
    print(f"  Unique sequences: mean={uniq_stats['mean']}, median={uniq_stats['median']}")

    return df_stats


if __name__ == "__main__":
    compute_length_stats()
