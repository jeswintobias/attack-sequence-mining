"""
Part 3 – Order-Signal Permutation Test.

Goal: Test whether the ordering of techniques within sighting lists carries
statistically non-random signal, or could plausibly be arbitrary.

Method:
  1. From the TRAIN sequences only, take the 30 most frequent techniques.
  2. Count adjacent transitions n(A→B) between consecutive positions.
  3. For every unordered pair {A,B} with n(A→B)+n(B→A) >= 20, compute
     asymmetry = |n(A→B)-n(B→A)| / (n(A→B)+n(B→A)).
  4. Real statistic = mean asymmetry over qualifying pairs.
  5. Shuffle items inside every sequence 200 times and recompute.
  6. Empirical p-value = (1 + #shuffled >= real) / 201.

References:
  - Transition asymmetry concept follows the directional analysis in
    Li et al. (DeepAG, IEEE TDSC 2023) who verify that attack paths have
    non-reversible temporal order.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import config


def compute_asymmetry_stat(sequences: list, top_techs: set, min_total: int = 20):
    """
    Compute mean asymmetry over unordered pairs {A,B} among top_techs
    where n(A→B) + n(B→A) >= min_total.

    Returns: (mean_asymmetry, n_qualifying_pairs, pair_details)
    """
    bigram_counter = Counter()
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            if a in top_techs and b in top_techs and a != b:
                bigram_counter[(a, b)] += 1

    # Build unordered pair asymmetries
    seen_pairs = set()
    asymmetries = []
    pair_details = []
    for (a, b), c_ab in bigram_counter.items():
        pair_key = tuple(sorted([a, b]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        c_ba = bigram_counter.get((b, a), 0)
        total = c_ab + c_ba
        if total >= min_total:
            asym = abs(c_ab - c_ba) / total
            asymmetries.append(asym)
            pair_details.append({
                "A": a, "B": b,
                "n_AB": c_ab, "n_BA": c_ba,
                "total": total,
                "asymmetry": round(asym, 4),
            })

    if len(asymmetries) == 0:
        return 0.0, 0, []
    return float(np.mean(asymmetries)), len(asymmetries), pair_details


def run_order_signal_test(
    train_seqs_df: pd.DataFrame = None,
    results_dir: Path = None,
    n_shuffles: int = 200,
):
    """Run the permutation test on train sequences only."""
    if results_dir is None:
        results_dir = config.EDA_RESULTS_DIR
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load train sequences if not provided
    if train_seqs_df is None:
        train_seqs_df = pd.read_csv(
            config.BASE_DIR / "data" / "processed" / "train_sequences.csv"
        )
        train_seqs_df["seq_tuple"] = train_seqs_df["sequence"].apply(
            lambda s: tuple(eval(s))
        )

    # Expand by dup_count for correct weighting
    train_seqs = []
    for _, r in train_seqs_df.iterrows():
        for _ in range(r["dup_count"]):
            train_seqs.append(r["seq_tuple"])

    print(f"Order-signal test on {len(train_seqs):,} train sequences "
          f"({len(train_seqs_df)} unique)")

    # Top 30 techniques by frequency
    all_techs = [t for seq in train_seqs for t in seq]
    tech_counts = Counter(all_techs)
    top30 = set(t for t, _ in tech_counts.most_common(30))
    print(f"Top 30 techniques selected (most frequent in train)")

    # Real statistic
    real_stat, n_pairs, pair_details = compute_asymmetry_stat(
        train_seqs, top30, min_total=20
    )
    print(f"Real mean asymmetry: {real_stat:.4f} over {n_pairs} qualifying pairs")

    # Save pair details
    pd.DataFrame(pair_details).to_csv(
        fig_dir / "14_order_signal_pair_details.csv", index=False
    )

    # Shuffled distribution
    rng = np.random.RandomState(config.RANDOM_SEED)
    shuffled_stats = []
    for i in range(n_shuffles):
        shuffled_seqs = []
        for seq in train_seqs:
            perm = list(seq)
            rng.shuffle(perm)
            shuffled_seqs.append(tuple(perm))
        s, _, _ = compute_asymmetry_stat(shuffled_seqs, top30, min_total=20)
        shuffled_stats.append(s)
        if (i + 1) % 50 == 0:
            print(f"  Completed {i+1}/{n_shuffles} shuffles")

    shuffled_arr = np.array(shuffled_stats)
    shuf_mean = float(shuffled_arr.mean())
    p2_5 = float(np.percentile(shuffled_arr, 2.5))
    p97_5 = float(np.percentile(shuffled_arr, 97.5))
    n_ge = int((shuffled_arr >= real_stat).sum())
    p_value = (1 + n_ge) / (n_shuffles + 1)

    print(f"\nShuffled mean: {shuf_mean:.4f}")
    print(f"95% CI: [{p2_5:.4f}, {p97_5:.4f}]")
    print(f"Empirical p-value: {p_value:.4f}")
    print(f"  ({n_ge} of {n_shuffles} shuffled values >= real value)")

    # Save results CSV
    result_df = pd.DataFrame([{
        "real_mean_asymmetry": round(real_stat, 6),
        "n_qualifying_pairs": n_pairs,
        "shuffled_mean": round(shuf_mean, 6),
        "shuffled_p2.5": round(p2_5, 6),
        "shuffled_p97.5": round(p97_5, 6),
        "n_shuffled_ge_real": n_ge,
        "p_value": round(p_value, 6),
        "n_shuffles": n_shuffles,
    }])
    result_df.to_csv(fig_dir / "14_order_signal_results.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(shuffled_arr, bins=30, color="#4e79a7", edgecolor="black",
            alpha=0.75, label="Shuffled distribution")
    ax.axvline(real_stat, color="red", linewidth=2, linestyle="--",
               label=f"Real = {real_stat:.4f}")
    ax.axvline(p2_5, color="gray", linewidth=1, linestyle=":",
               label=f"2.5% = {p2_5:.4f}")
    ax.axvline(p97_5, color="gray", linewidth=1, linestyle=":",
               label=f"97.5% = {p97_5:.4f}")
    ax.set_xlabel("Mean asymmetry statistic")
    ax.set_ylabel("Count (of 200 shuffles)")
    ax.set_title(
        f"Order-Signal Permutation Test (train sequences, dup-weighted)\n"
        f"Real = {real_stat:.4f}, Shuffled mean = {shuf_mean:.4f}, "
        f"p = {p_value:.4f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "14_order_signal_test.png", dpi=300)
    plt.close(fig)
    print(f"Saved 14_order_signal_test.png")

    return {
        "real_stat": real_stat,
        "shuffled_mean": shuf_mean,
        "p2_5": p2_5,
        "p97_5": p97_5,
        "p_value": p_value,
        "n_pairs": n_pairs,
    }


if __name__ == "__main__":
    run_order_signal_test()
