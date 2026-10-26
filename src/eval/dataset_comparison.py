"""Structural comparison of sightings_v2 and CPTC-2018.

Run from the repository root:
  python src/eval/dataset_comparison.py

The two datasets produce very different scores with identical models (0.539
against 0.766 sequence MRR). Before attributing that to execution order versus
reported order, the alternative explanation has to be ruled out: that CPTC is
simply a less ambiguous prediction problem. This script measures the properties
that drive predictability on both, so the comparison can be qualified honestly.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "eval")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config

OUT_DIR = config.RESULTS_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _entropy(counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * log(c / total, 2) for c in counter.values() if c > 0)


def profile(examples: pd.DataFrame, name: str) -> dict:
    """Properties that govern how predictable a next-symbol task is."""
    sequences = examples.drop_duplicates("seq_id")["full_seq"] if "seq_id" in examples \
        else examples["full_seq"].drop_duplicates()
    lengths = np.array([len(s) for s in sequences])
    labels = examples["next"].astype(str)
    label_counts = labels.value_counts()
    share = label_counts / label_counts.sum()

    # Branching: distinct successors observed per context, the direct driver of
    # how much irreducible ambiguity the task carries.
    branching = {}
    for depth, key in ((1, lambda p: (p[-1],)), (2, lambda p: tuple(p[-2:]))):
        table = defaultdict(Counter)
        for prefix, nxt in zip(examples["prefix"], labels):
            table[key(prefix)][nxt] += 1
        sizes = np.array([sum(c.values()) for c in table.values()])
        distinct = np.array([len(c) for c in table.values()])
        reused = sizes >= 2
        weights = sizes / sizes.sum()
        branching[depth] = {
            "contexts": len(table),
            "reuse_rate": float(reused.mean()),
            "mean_successors_when_reused": float(distinct[reused].mean()) if reused.any() else 0.0,
            "weighted_conditional_entropy_bits": float(
                sum(w * _entropy(c) for w, c in zip(weights, table.values()))),
        }

    repeats = float(np.mean([row["next"] in set(row["prefix"])
                             for _, row in examples.iterrows()]))
    return {
        "dataset": name,
        "sequences": int(len(sequences)),
        "examples": int(len(examples)),
        "label_classes": int(labels.nunique()),
        "mean_length": round(float(lengths.mean()), 2),
        "median_length": int(np.median(lengths)),
        "max_length": int(lengths.max()),
        "top1_label_share": round(float(share.iloc[0]), 4),
        "top10_label_share": round(float(share.head(10).sum()), 4),
        "label_entropy_bits": round(_entropy(Counter(labels)), 3),
        "label_entropy_normalised": round(
            _entropy(Counter(labels)) / log(labels.nunique(), 2), 3),
        "classes_under_5_examples": int((label_counts < 5).sum()),
        "target_repeats_prefix": round(repeats, 4),
        "last1_contexts": branching[1]["contexts"],
        "last1_reuse": round(branching[1]["reuse_rate"], 3),
        "last1_mean_successors": round(branching[1]["mean_successors_when_reused"], 2),
        "last1_cond_entropy_bits": round(branching[1]["weighted_conditional_entropy_bits"], 3),
        "last2_contexts": branching[2]["contexts"],
        "last2_reuse": round(branching[2]["reuse_rate"], 3),
        "last2_mean_successors": round(branching[2]["mean_successors_when_reused"], 2),
        "last2_cond_entropy_bits": round(branching[2]["weighted_conditional_entropy_bits"], 3),
    }


def main():
    from cv import load_development_examples

    rows = [profile(load_development_examples(), "sightings_v2 (reported order)")]
    for suffix, label in (("stage", "CPTC-2018 stage only"),
                          ("service", "CPTC-2018 stage|service")):
        path = config.PROCESSED_DATA_DIR / f"cptc_examples_{suffix}.pkl"
        if path.exists():
            rows.append(profile(pd.read_pickle(path), label))
        else:
            print(f"missing {path}; run python src/core/cptc_loader.py")

    frame = pd.DataFrame(rows).set_index("dataset").T
    print(frame.to_string())
    frame.to_csv(OUT_DIR / "dataset_structural_comparison.csv")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print("  Conditional entropy is the headline number: it is the average bits of")
    print("  uncertainty left about the next symbol once the context is known, so")
    print("  it measures task difficulty independently of how many classes exist.")
    print("  A lower value means an inherently easier prediction problem.")
    print(f"\nSaved to {OUT_DIR / 'dataset_structural_comparison.csv'}")


if __name__ == "__main__":
    main()
