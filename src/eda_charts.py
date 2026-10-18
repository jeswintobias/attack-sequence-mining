"""
Stage A2 – Comprehensive EDA Charts.

Reads data/sightings_v2_public.csv entirely in-memory (raw file immutable).
All figures saved at 300 dpi in results/figures/ with companion CSVs.
"""

import ast
import itertools
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as sp_stats

import sys
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# ──────────────────────────────────────────────
# Global settings
# ──────────────────────────────────────────────
FIG_DIR = config.RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def _save(fig, name):
    """Save figure to results/figures/<name>.png and close."""
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png")


def _save_csv(df, name):
    """Save companion CSV to results/figures/<name>.csv."""
    df.to_csv(FIG_DIR / f"{name}.csv")
    print(f"  Saved {name}.csv")


# ──────────────────────────────────────────────
# Load and parse once
# ──────────────────────────────────────────────
def load_parsed():
    """Return the full DataFrame with parsed columns. Raw file stays immutable."""
    print("Loading data/sightings_v2_public.csv …")
    df = pd.read_csv(config.SIGHTINGS_DATA_PATH)
    df["tid_parsed"] = df["tid"].apply(ast.literal_eval)
    df["seq_len"] = df["tid_parsed"].apply(len)
    df["date_parsed"] = pd.to_datetime(
        df["sighting_date"], format="%m/%d/%y", errors="coerce"
    )
    df["year"] = df["date_parsed"].dt.year
    # Flag bad dates (< 2000 or > 2026)
    df["date_valid"] = df["year"].between(2000, 2026, inclusive="both")
    print(f"  Rows: {len(df):,}  |  Bad dates: {(~df['date_valid']).sum()}")
    return df


# ════════════════════════════════════════════════
# Chart 1 – Row share by length class
# ════════════════════════════════════════════════
def chart_01_length_class_share(df):
    print("\n[1] Row share by sequence-length class")
    bins = pd.cut(
        df["seq_len"],
        bins=[0, 1, 2, 3, float("inf")],
        labels=["1", "2", "3", "4+"],
    )
    counts = bins.value_counts().sort_index()
    pcts = (counts / len(df) * 100).round(2)
    tbl = pd.DataFrame({"count": counts, "pct": pcts})

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(tbl.index, tbl["pct"], color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"])
    ax.set_xlabel("Sequence-length class")
    ax.set_ylabel("Share of rows (%)")
    ax.set_title(
        f"Row share by sequence-length class (N = {len(df):,} rows)\n"
        f"Singletons dominate at {pcts.iloc[0]:.1f}%; "
        f"length ≥ 4 accounts for {pcts.iloc[3]:.1f}%"
    )
    for b, v in zip(bars, tbl["pct"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(tbl["pct"]) * 1.15)
    _save(fig, "01_length_class_share")
    _save_csv(tbl, "01_length_class_share")


# ════════════════════════════════════════════════
# Chart 2 – Length histogram for length ≥ 4
# ════════════════════════════════════════════════
def chart_02_length_hist_ge4(df):
    print("\n[2] Length histogram (length ≥ 4, log-y)")
    lens = df.loc[df["seq_len"] >= 4, "seq_len"]
    mean_l, med_l = lens.mean(), lens.median()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lens, bins=range(4, int(lens.max()) + 2), color="#4e79a7",
            edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.axvline(mean_l, color="red", ls="--", lw=1.5, label=f"Mean = {mean_l:.1f}")
    ax.axvline(med_l, color="green", ls="--", lw=1.5, label=f"Median = {med_l:.0f}")
    ax.legend()
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Row count (log scale)")
    ax.set_title(
        f"Length distribution of multi-stage rows (N = {len(lens):,} rows, uncollapsed)\n"
        f"Mean {mean_l:.1f}, Median {med_l:.0f}, Max {int(lens.max())}"
    )
    _save(fig, "02_length_hist_ge4")

    # Companion CSV
    vc = lens.value_counts().sort_index()
    _save_csv(vc.to_frame("count"), "02_length_hist_ge4")


# ════════════════════════════════════════════════
# Chart 3 – Top 20 techniques: all rows vs long
# ════════════════════════════════════════════════
def chart_03_top_techniques(df):
    print("\n[3] Top 20 techniques – all rows vs length ≥ 4")
    all_flat = [t for seq in df["tid_parsed"] for t in seq]
    all_counts = pd.Series(Counter(all_flat))

    long_flat = [t for seq in df.loc[df["seq_len"] >= 4, "tid_parsed"] for t in seq]
    long_counts = pd.Series(Counter(long_flat))

    top20 = all_counts.nlargest(20).index
    tbl = pd.DataFrame({
        "all_rows": all_counts.reindex(top20).fillna(0).astype(int),
        "long_seqs": long_counts.reindex(top20).fillna(0).astype(int),
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(top20))
    w = 0.38
    ax.bar(x - w / 2, tbl["all_rows"], w, label="All rows", color="#4e79a7")
    ax.bar(x + w / 2, tbl["long_seqs"], w, label="Length ≥ 4", color="#e15759")
    ax.set_xticks(x)
    ax.set_xticklabels(top20, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Occurrence count")
    ax.set_xlabel("MITRE ATT&CK technique")
    ax.set_title("Top 20 techniques: all rows vs multi-stage rows (uncollapsed)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "03_top20_techniques")
    _save_csv(tbl, "03_top20_techniques")


# ════════════════════════════════════════════════
# Chart 4 – Rank vs frequency (Zipf), log-log
# ════════════════════════════════════════════════
def chart_04_zipf(df):
    print("\n[4] Rank-frequency (Zipf), log-log")
    all_flat = [t for seq in df["tid_parsed"] for t in seq]
    freq = pd.Series(Counter(all_flat)).sort_values(ascending=False).values
    ranks = np.arange(1, len(freq) + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ranks, freq, s=15, alpha=0.7, color="#4e79a7")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Frequency (log)")
    ax.set_title(
        f"Technique rank vs frequency (N_unique = {len(freq)} techniques, all rows)\n"
        f"Top technique appears {freq[0]:,}×; tail has {(freq < 100).sum()} techniques < 100"
    )
    _save(fig, "04_zipf_rank_freq")

    tbl = pd.DataFrame({"rank": ranks, "frequency": freq})
    _save_csv(tbl, "04_zipf_rank_freq")


# ════════════════════════════════════════════════
# Chart 5 – Sightings per month (clean dates)
# ════════════════════════════════════════════════
def chart_05_monthly_volume(df):
    print("\n[5] Monthly sighting volume (clean dates)")
    clean = df[df["date_valid"]].copy()
    monthly = clean.set_index("date_parsed").resample("ME").size()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly.values, marker="o", markersize=3,
            color="#4e79a7", linewidth=1.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sightings per month")
    ax.set_title(
        f"Monthly sighting volume ({len(clean):,} rows, "
        f"14 bad-date rows excluded, counting rows)"
    )
    ax.grid(axis="y", ls="--", alpha=0.5)
    fig.autofmt_xdate()
    _save(fig, "05_monthly_volume")

    tbl = monthly.to_frame("count")
    tbl.index.name = "month"
    _save_csv(tbl, "05_monthly_volume")


# ════════════════════════════════════════════════
# Chart 6 – Box plot of length by year
# ════════════════════════════════════════════════
def chart_06_length_by_year(df):
    print("\n[6] Box-plot of sequence length by year")
    clean = df[df["date_valid"]].copy()
    years = sorted(clean["year"].dropna().unique().astype(int))

    fig, ax = plt.subplots(figsize=(8, 4))
    data_by_year = [clean.loc[clean["year"] == y, "seq_len"].values for y in years]
    bp = ax.boxplot(data_by_year, tick_labels=[str(y) for y in years],
                    patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4e79a7")
        patch.set_alpha(0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Sequence length (log scale)")
    ax.set_yscale("log")
    ax.set_title("Sequence-length distribution by year (rows, log scale, outliers shown)")
    ax.grid(axis="y", ls="--", alpha=0.4)
    _save(fig, "06_length_by_year")

    stats_list = []
    for y, d in zip(years, data_by_year):
        s = pd.Series(d)
        stats_list.append({"year": y, "median": s.median(), "mean": round(s.mean(), 2),
                           "q25": s.quantile(0.25), "q75": s.quantile(0.75), "n": len(s)})
    _save_csv(pd.DataFrame(stats_list).set_index("year"), "06_length_by_year")


# ════════════════════════════════════════════════
# Chart 7 – Top 10 most repeated sequences
# ════════════════════════════════════════════════
def chart_07_top_repeated_seqs(df):
    print("\n[7] Top 10 most repeated sequences")
    ge4 = df[df["seq_len"] >= 4]
    seq_counts = ge4["tid"].value_counts()
    total_rows = len(ge4)
    unique_seqs = len(seq_counts)

    top10 = seq_counts.head(10)
    labels = [s[:60] + "…" if len(s) > 60 else s for s in top10.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(top10)), top10.values, color="#59a14f")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Row count (duplicates)")
    ax.set_title(
        f"Top 10 most repeated sequences (length ≥ 4, counting rows)\n"
        f"Total rows = {total_rows:,}; unique sequences = {unique_seqs:,}; "
        f"duplication ratio = {total_rows / unique_seqs:.1f}×"
    )
    fig.tight_layout()
    _save(fig, "07_top10_repeated_seqs")

    tbl = top10.to_frame("row_count")
    tbl.insert(0, "label", labels)
    _save_csv(tbl, "07_top10_repeated_seqs")


# ════════════════════════════════════════════════
# Chart 8 – Most common first & last techniques
# ════════════════════════════════════════════════
def chart_08_first_last(df):
    print("\n[8] Most common first and last techniques")
    ge4 = df[df["seq_len"] >= 4]
    firsts = ge4["tid_parsed"].apply(lambda x: x[0])
    lasts = ge4["tid_parsed"].apply(lambda x: x[-1])

    top_first = firsts.value_counts().head(10)
    top_last = lasts.value_counts().head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.barh(top_first.index[::-1], top_first.values[::-1], color="#4e79a7")
    ax1.set_xlabel("Count")
    ax1.set_title("Top 10 FIRST techniques (length ≥ 4)")

    ax2.barh(top_last.index[::-1], top_last.values[::-1], color="#e15759")
    ax2.set_xlabel("Count")
    ax2.set_title("Top 10 LAST techniques (length ≥ 4)")

    fig.suptitle(
        f"First vs last technique position (N = {len(ge4):,} rows, uncollapsed)",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "08_first_last_techniques")

    tbl = pd.DataFrame({
        "first_technique": top_first.index, "first_count": top_first.values,
    }).merge(
        pd.DataFrame({
            "last_technique": top_last.index, "last_count": top_last.values,
        }),
        left_index=True, right_index=True, how="outer"
    )
    _save_csv(tbl, "08_first_last_techniques")


# ════════════════════════════════════════════════
# Chart 9 – Co-occurrence lift heatmap (top 25)
# ════════════════════════════════════════════════
def chart_09_cooccurrence_lift(df):
    print("\n[9] Co-occurrence lift heatmap (top 25 techniques)")
    ge4 = df[df["seq_len"] >= 4]
    n_txn = len(ge4)

    # Build item-sets from each row
    item_sets = ge4["tid_parsed"].apply(set)

    # Top 25 by global frequency within length >= 4
    all_flat = [t for s in ge4["tid_parsed"] for t in s]
    top25 = pd.Series(Counter(all_flat)).nlargest(25).index.tolist()

    # Support of each item
    sup = {}
    for t in top25:
        sup[t] = item_sets.apply(lambda s: t in s).sum() / n_txn

    # Pairwise co-occurrence support
    lift_mat = pd.DataFrame(np.ones((25, 25)), index=top25, columns=top25)
    for i, a in enumerate(top25):
        for j, b in enumerate(top25):
            if i == j:
                lift_mat.iloc[i, j] = np.nan
                continue
            co_sup = item_sets.apply(lambda s: a in s and b in s).sum() / n_txn
            denom = sup[a] * sup[b]
            lift_mat.iloc[i, j] = co_sup / denom if denom > 0 else 0

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(lift_mat.values.astype(float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(25))
    ax.set_yticks(range(25))
    ax.set_xticklabels(top25, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(top25, fontsize=7)
    fig.colorbar(im, ax=ax, label="Lift")
    ax.set_title(
        f"Pairwise co-occurrence lift (top 25 techniques, N = {n_txn:,} rows)\n"
        "Lift > 1 ⇒ positive association; lift < 1 ⇒ negative association"
    )
    fig.tight_layout()
    _save(fig, "09_cooccurrence_lift")
    _save_csv(lift_mat, "09_cooccurrence_lift")


# ════════════════════════════════════════════════
# Chart 10 – Transition count & asymmetry heatmaps
# ════════════════════════════════════════════════
def chart_10_transition_heatmaps(df):
    print("\n[10] Transition count & asymmetry heatmaps (top 20)")
    ge4 = df[df["seq_len"] >= 4]

    # Count bigram transitions
    bigram_counter = Counter()
    for seq in ge4["tid_parsed"]:
        for a, b in zip(seq[:-1], seq[1:]):
            bigram_counter[(a, b)] += 1

    all_flat = [t for s in ge4["tid_parsed"] for t in s]
    top20 = pd.Series(Counter(all_flat)).nlargest(20).index.tolist()

    trans_mat = pd.DataFrame(0, index=top20, columns=top20, dtype=int)
    for (a, b), c in bigram_counter.items():
        if a in top20 and b in top20:
            trans_mat.loc[a, b] = c

    asym_mat = trans_mat - trans_mat.T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    im1 = ax1.imshow(trans_mat.values, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(20))
    ax1.set_yticks(range(20))
    ax1.set_xticklabels(top20, rotation=60, ha="right", fontsize=7)
    ax1.set_yticklabels(top20, fontsize=7)
    ax1.set_title("Transition count (row → col)")
    ax1.set_xlabel("To technique")
    ax1.set_ylabel("From technique")
    fig.colorbar(im1, ax=ax1, label="Count", shrink=0.7)

    vmax = max(abs(asym_mat.values.min()), abs(asym_mat.values.max()))
    im2 = ax2.imshow(asym_mat.values, cmap="RdBu_r", aspect="auto",
                     vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(20))
    ax2.set_yticks(range(20))
    ax2.set_xticklabels(top20, rotation=60, ha="right", fontsize=7)
    ax2.set_yticklabels(top20, fontsize=7)
    ax2.set_title("Asymmetry: count(A→B) − count(B→A)\nRed = A→B dominates; Blue = B→A dominates")
    ax2.set_xlabel("To technique")
    ax2.set_ylabel("From technique")
    fig.colorbar(im2, ax=ax2, label="Asymmetry", shrink=0.7)

    fig.suptitle("Bigram transition analysis (top 20 techniques, length ≥ 4 rows)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "10_transition_heatmaps")

    _save_csv(trans_mat, "10_transition_count")
    _save_csv(asym_mat, "10_transition_asymmetry")


# ════════════════════════════════════════════════
# Chart 11 – Next-technique label distribution
# ════════════════════════════════════════════════
def chart_11_next_label_dist(df):
    print("\n[11] Next-technique label distribution")
    ge4 = df[df["seq_len"] >= 4]

    # Generate all (prefix, next) pairs from length >= 4 rows
    next_labels = []
    for seq in ge4["tid_parsed"]:
        for k in range(2, len(seq)):
            next_labels.append(seq[k])

    next_counts = pd.Series(Counter(next_labels)).sort_values(ascending=False)
    total = next_counts.sum()
    majority_class = next_counts.index[0]
    majority_acc = next_counts.iloc[0] / total * 100

    top20 = next_counts.head(20)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top20)), top20.values, color="#4e79a7")
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels(top20.index, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Count as next-technique label")
    ax.set_xlabel("Technique")
    ax.set_title(
        f"Next-technique label distribution (top 20 of {len(next_counts)} unique, counting rows)\n"
        f"Majority class = {majority_class} (from all rows, not train-only)"
    )
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "11_next_label_dist")

    tbl = next_counts.to_frame("count")
    tbl["pct"] = (tbl["count"] / total * 100).round(3)
    _save_csv(tbl, "11_next_label_dist")

    return majority_class, majority_acc


# ════════════════════════════════════════════════
# Bonus 12 – Subtechnique depth analysis
# ════════════════════════════════════════════════
def chart_12_subtechnique_depth(df):
    print("\n[12] Subtechnique depth analysis")
    all_flat = [t for seq in df["tid_parsed"] for t in seq]
    depths = pd.Series([t.count(".") for t in all_flat])
    depth_counts = depths.value_counts().sort_index()
    depth_labels = {0: "Parent (Txxx)", 1: "Sub (Txxx.yyy)"}

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([depth_labels.get(i, f"Depth {i}") for i in depth_counts.index],
                  depth_counts.values, color=["#4e79a7", "#e15759"])
    for b, v in zip(bars, depth_counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total occurrences")
    ax.set_title("Parent vs sub-technique usage across all sighting rows")
    _save(fig, "12_subtechnique_depth")
    _save_csv(depth_counts.to_frame("count"), "12_subtechnique_depth")


# ════════════════════════════════════════════════
# Bonus 13 – Sequence diversity: unique/total ratio per length
# ════════════════════════════════════════════════
def chart_13_diversity_ratio(df):
    print("\n[13] Sequence diversity ratio by length")
    ge4 = df[df["seq_len"] >= 4]
    grouped = ge4.groupby("seq_len")["tid"].agg(["count", "nunique"])
    grouped["diversity_ratio"] = grouped["nunique"] / grouped["count"]
    grouped = grouped[grouped["count"] >= 5]  # avoid noisy singletons

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grouped.index, grouped["diversity_ratio"], marker="o",
            color="#4e79a7", linewidth=1.5)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Diversity ratio (unique / total)")
    ax.set_title(
        "Sequence diversity ratio by length (length ≥ 4, rows vs unique sequences)\n"
        "Higher = more distinct attack patterns at that length"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(ls="--", alpha=0.4)
    _save(fig, "13_diversity_ratio")
    _save_csv(grouped, "13_diversity_ratio")


# ════════════════════════════════════════════════
# Length stats CSV
# ════════════════════════════════════════════════
def write_length_stats(df):
    print("\nWriting results/length_stats.csv")
    lens = df["seq_len"]
    stats = {
        "mean": round(lens.mean(), 4),
        "median": lens.median(),
        "std": round(lens.std(), 4),
        "skew": round(lens.skew(), 4),
        "min": lens.min(),
        "q25": lens.quantile(0.25),
        "q50": lens.quantile(0.50),
        "q75": lens.quantile(0.75),
        "max": lens.max(),
        "n": len(lens),
    }
    out = pd.DataFrame([stats])
    path = config.RESULTS_DIR / "length_stats.csv"
    out.to_csv(path, index=False)
    print(f"  Saved {path}")


# ════════════════════════════════════════════════
# Main entry
# ════════════════════════════════════════════════
def main():
    df = load_parsed()

    chart_01_length_class_share(df)
    chart_02_length_hist_ge4(df)
    chart_03_top_techniques(df)
    chart_04_zipf(df)
    chart_05_monthly_volume(df)
    chart_06_length_by_year(df)
    chart_07_top_repeated_seqs(df)
    chart_08_first_last(df)
    chart_09_cooccurrence_lift(df)
    chart_10_transition_heatmaps(df)
    maj_cls, maj_acc = chart_11_next_label_dist(df)
    chart_12_subtechnique_depth(df)
    chart_13_diversity_ratio(df)
    write_length_stats(df)

    print("\n" + "=" * 60)
    print("All 13 figures + companion CSVs + length_stats.csv saved.")
    print(f"Majority-class baseline accuracy: {maj_acc:.2f}% ({maj_cls})")
    print("=" * 60)


if __name__ == "__main__":
    main()
