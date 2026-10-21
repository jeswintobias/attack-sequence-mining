"""
Task 1 – Order-Signal Control Test.

Extends the original permutation test with two robustness checks:
  a) Drop sequences whose technique IDs (the numeric part) are fully
     ascending or fully descending (they could be artefacts of ID-sorted
     reporting). Re-run the 200-shuffle permutation test on the remaining
     sequences.
  b) Among "strong" pairs (asymmetry > 0.5, qualifying >= 20), report the
     share whose dominant direction follows ascending technique-ID order.
     If order were merely ascending-ID bias, this share would be ~1.0.

Design choices (for viva):
  - Smallest possible p-value is 1/(n_shuffles+1) = 1/201 ≈ 0.00498.
  - We do NOT claim the order is chronological.

References:
  - Directional analysis approach follows Li et al. (DeepAG, IEEE TDSC 2023).
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


def extract_id_number(tech_id: str) -> int:
    """Extract the numeric portion of a technique ID, e.g. T1059 -> 1059."""
    return int(tech_id.replace("T", ""))


def is_monotonic(seq: tuple) -> bool:
    """Return True if the numeric IDs in seq are fully ascending or descending."""
    nums = [extract_id_number(t) for t in seq]
    ascending = all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))
    descending = all(nums[i] >= nums[i + 1] for i in range(len(nums) - 1))
    return ascending or descending


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
            # Determine dominant direction
            if c_ab >= c_ba:
                dom_fwd, dom_rev = a, b
            else:
                dom_fwd, dom_rev = b, a
            asymmetries.append(asym)
            pair_details.append({
                "A": a, "B": b,
                "n_AB": c_ab, "n_BA": c_ba,
                "total": total,
                "asymmetry": round(asym, 4),
                "dominant_from": dom_fwd,
                "dominant_to": dom_rev,
                "dom_follows_ascending_id": int(
                    extract_id_number(dom_fwd) < extract_id_number(dom_rev)
                ),
            })

    if len(asymmetries) == 0:
        return 0.0, 0, []
    return float(np.mean(asymmetries)), len(asymmetries), pair_details


def run_order_signal_control(
    results_dir: Path = None,
    n_shuffles: int = 200,
):
    """Run the controlled permutation test after dropping monotonic sequences."""
    if results_dir is None:
        results_dir = config.EDA_RESULTS_DIR
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load train sequences
    train_seqs_df = pd.read_csv(
        config.BASE_DIR / "data" / "processed" / "train_sequences.csv"
    )
    train_seqs_df["seq_tuple"] = train_seqs_df["sequence"].apply(
        lambda s: tuple(eval(s))
    )

    # Count and drop monotonic (ascending/descending by ID) sequences
    train_seqs_df["is_monotonic"] = train_seqs_df["seq_tuple"].apply(is_monotonic)
    n_mono = train_seqs_df["is_monotonic"].sum()
    total_unique = len(train_seqs_df)
    n_mono_weighted = int(
        train_seqs_df.loc[train_seqs_df["is_monotonic"], "dup_count"].sum()
    )
    total_weighted = int(train_seqs_df["dup_count"].sum())
    print(f"Monotonic sequences (ascending or descending ID order):")
    print(f"  {n_mono}/{total_unique} unique "
          f"({n_mono / total_unique * 100:.1f}%)")
    print(f"  {n_mono_weighted}/{total_weighted} dup-weighted "
          f"({n_mono_weighted / total_weighted * 100:.1f}%)")

    # Filter to non-monotonic
    filtered_df = train_seqs_df[~train_seqs_df["is_monotonic"]].copy()
    print(f"\nAfter dropping monotonic: {len(filtered_df)} unique, "
          f"{int(filtered_df['dup_count'].sum())} dup-weighted")

    # Expand by dup_count for correct weighting
    train_seqs = []
    for _, r in filtered_df.iterrows():
        for _ in range(r["dup_count"]):
            train_seqs.append(r["seq_tuple"])

    print(f"\nOrder-signal control test on {len(train_seqs):,} "
          f"non-monotonic train sequences ({len(filtered_df)} unique)")

    # Top 30 techniques by frequency in filtered set
    all_techs = [t for seq in train_seqs for t in seq]
    tech_counts = Counter(all_techs)
    top30 = set(t for t, _ in tech_counts.most_common(30))

    # Real statistic
    real_stat, n_pairs, pair_details = compute_asymmetry_stat(
        train_seqs, top30, min_total=20
    )
    print(f"Real mean asymmetry: {real_stat:.4f} over {n_pairs} qualifying pairs")

    # Save pair details
    df_pairs = pd.DataFrame(pair_details)
    df_pairs.to_csv(
        fig_dir / "15_order_control_pair_details.csv", index=False
    )

    # Report share of strong pairs following ascending ID order
    strong_pairs = df_pairs[df_pairs["asymmetry"] > 0.5].copy()
    n_strong = len(strong_pairs)
    if n_strong > 0:
        n_asc_id = int(strong_pairs["dom_follows_ascending_id"].sum())
        asc_share = n_asc_id / n_strong
    else:
        n_asc_id = 0
        asc_share = float("nan")
    print(f"\nStrong pairs (asymmetry > 0.5): {n_strong}")
    print(f"  Following ascending ID order: {n_asc_id}/{n_strong} "
          f"= {asc_share:.4f}")

    # Shuffled distribution (200 shuffles → min p = 1/201)
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
            print(f"  Completed {i + 1}/{n_shuffles} shuffles")

    shuffled_arr = np.array(shuffled_stats)
    shuf_mean = float(shuffled_arr.mean())
    p2_5 = float(np.percentile(shuffled_arr, 2.5))
    p97_5 = float(np.percentile(shuffled_arr, 97.5))
    n_ge = int((shuffled_arr >= real_stat).sum())
    p_value = (1 + n_ge) / (n_shuffles + 1)

    print(f"\nShuffled mean: {shuf_mean:.4f}")
    print(f"95% CI: [{p2_5:.4f}, {p97_5:.4f}]")
    print(f"Empirical p-value: {p_value:.4f} "
          f"(smallest possible: {1 / (n_shuffles + 1):.4f})")
    print(f"  ({n_ge} of {n_shuffles} shuffled values >= real value)")

    # Save results CSV
    result_df = pd.DataFrame([{
        "variant": "control_no_monotonic",
        "n_unique_seqs_used": len(filtered_df),
        "n_monotonic_dropped_unique": n_mono,
        "n_monotonic_dropped_weighted": n_mono_weighted,
        "real_mean_asymmetry": round(real_stat, 6),
        "n_qualifying_pairs": n_pairs,
        "n_strong_pairs_gt05": n_strong,
        "strong_pairs_ascending_id_share": round(asc_share, 4),
        "shuffled_mean": round(shuf_mean, 6),
        "shuffled_p2.5": round(p2_5, 6),
        "shuffled_p97.5": round(p97_5, 6),
        "n_shuffled_ge_real": n_ge,
        "p_value": round(p_value, 6),
        "n_shuffles": n_shuffles,
    }])
    result_df.to_csv(fig_dir / "15_order_control_results.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(shuffled_arr, bins=30, color="#59a14f", edgecolor="black",
            alpha=0.75, label="Shuffled distribution")
    ax.axvline(real_stat, color="red", linewidth=2, linestyle="--",
               label=f"Real = {real_stat:.4f}")
    ax.axvline(p2_5, color="gray", linewidth=1, linestyle=":",
               label=f"2.5% = {p2_5:.4f}")
    ax.axvline(p97_5, color="gray", linewidth=1, linestyle=":",
               label=f"97.5% = {p97_5:.4f}")
    ax.set_xlabel("Mean asymmetry statistic")
    ax.set_ylabel(f"Count (of {n_shuffles} shuffles)")
    ax.set_title(
        f"Order-Signal Control (monotonic seqs dropped, dup-weighted)\n"
        f"Real = {real_stat:.4f}, Shuffled mean = {shuf_mean:.4f}, "
        f"p = {p_value:.4f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "15_order_control_test.png", dpi=300)
    plt.close(fig)
    print(f"\nSaved 15_order_control_test.png")

    return {
        "real_stat": real_stat,
        "shuffled_mean": shuf_mean,
        "p2_5": p2_5,
        "p97_5": p97_5,
        "p_value": p_value,
        "n_pairs": n_pairs,
        "n_strong": n_strong,
        "asc_share": asc_share,
        "n_mono_dropped": n_mono,
    }


if __name__ == "__main__":
    run_order_signal_control()
