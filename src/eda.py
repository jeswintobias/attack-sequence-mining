"""
Exploratory Data Analysis (EDA) Module.

Executes complete in-memory analysis of data/sightings_v2_public.csv without altering
the underlying raw file on disk.

Calculates:
1. Exact row count and data validation.
2. Safe AST parsing of stringified technique lists.
3. Timestamp auditing (%m/%d/%y), isolating valid vs. junk years.
4. Sequence cardinality & singleton isolation (length == 1 vs length >= 4).
5. Intra-list repetition analysis (verifying the 278 repeated technique rows).
6. Global unigram technique prior distribution (powering Bayesian back-off in Stage B).
7. Impact of subtechnique collapsing (Txxx.yyy -> Txxx).
8. Publication-quality multi-panel visualization saved to results/eda_overview.png.
"""

import ast
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def run_eda(data_path: Path, results_dir: Path) -> Dict[str, Any]:
    """Runs end-to-end in-memory EDA on the raw sightings dataset."""
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading raw sightings from: {data_path} ...")
    
    # 1. Ingestion (Raw file remains 100% immutable)
    df_raw = pd.read_csv(data_path)
    total_raw_rows = len(df_raw)
    print(f"Total raw rows loaded: {total_raw_rows:,}")
    
    # 2. Safe Parsing with ast.literal_eval
    print("Parsing technique string lists safely with ast.literal_eval...")
    df_raw["tid_parsed"] = df_raw["tid"].apply(ast.literal_eval)
    df_raw["seq_len"] = df_raw["tid_parsed"].apply(len)
    
    # 3. Date Parsing & Out-of-Range Year Auditing
    print("Auditing date timestamps with format %m/%d/%y...")
    df_raw["date_parsed"] = pd.to_datetime(df_raw["sighting_date"], format="%m/%d/%y", errors="coerce")
    df_raw["year"] = df_raw["date_parsed"].dt.year
    
    valid_year_mask = (df_raw["year"] >= 2000) & (df_raw["year"] <= 2026)
    junk_dates_df = df_raw[~valid_year_mask]
    junk_year_counts = junk_dates_df["year"].value_counts(dropna=False).to_dict()
    print(f"Flagged {len(junk_dates_df):,} rows with out-of-range or null years: {junk_year_counts}")
    
    # 4. Cardinality & Length Distribution
    length_counts = df_raw["seq_len"].value_counts().sort_index()
    singleton_count = (df_raw["seq_len"] == 1).sum()
    singleton_pct = (singleton_count / total_raw_rows) * 100
    
    df_ge4 = df_raw[df_raw["seq_len"] >= 4].copy()
    count_ge4 = len(df_ge4)
    unique_ge4 = df_ge4["tid"].nunique()
    
    print(f"Singletons (length == 1): {singleton_count:,} ({singleton_pct:.2f}%)")
    print(f"Sequences with length >= 4: {count_ge4:,} total ({unique_ge4:,} unique)")
    
    # 5. Internal Repetition Analysis
    def has_internal_repeats(tech_list):
        return len(tech_list) != len(set(tech_list))
    
    df_raw["has_repeats"] = df_raw["tid_parsed"].apply(has_internal_repeats)
    total_repeated_rows = df_raw["has_repeats"].sum()
    print(f"Rows with repeated techniques inside one list: {total_repeated_rows:,} (Expected: 278)")
    
    # 6. Global Technique Prior Calculation (All 1.04M rows including singletons)
    print("Computing global technique priors across all rows...")
    all_techniques_flat = [t for sublist in df_raw["tid_parsed"] for t in sublist]
    global_tech_counts = pd.Series(all_techniques_flat).value_counts()
    global_tech_priors = (global_tech_counts / len(all_techniques_flat)).to_frame(name="prior_probability")
    global_tech_priors["raw_count"] = global_tech_counts
    
    priors_csv_path = results_dir / "global_technique_priors.csv"
    global_tech_priors.to_csv(priors_csv_path)
    print(f"Saved global technique priors to: {priors_csv_path}")
    
    # Techniques in length >= 4 sequences
    seq_techniques_flat = [t for sublist in df_ge4["tid_parsed"] for t in sublist]
    seq_tech_counts = pd.Series(seq_techniques_flat).value_counts()
    
    # 7. Subtechnique Collapsing Analysis
    def collapse_list(tech_list):
        return [t.split(".")[0] for t in tech_list]
    
    all_parent_flat = [t.split(".")[0] for t in all_techniques_flat]
    unique_fine_grained = len(global_tech_counts)
    unique_parents = pd.Series(all_parent_flat).nunique()
    print(f"Unique techniques: {unique_fine_grained:,} fine-grained -> {unique_parents:,} parent techniques")
    
    # 8. Save EDA Summary Table
    eda_summary_records = [
        {"metric": "total_raw_rows", "value": total_raw_rows},
        {"metric": "singleton_rows", "value": singleton_count},
        {"metric": "singleton_percentage", "value": round(singleton_pct, 2)},
        {"metric": "rows_length_2_to_3", "value": int(((df_raw["seq_len"] >= 2) & (df_raw["seq_len"] <= 3)).sum())},
        {"metric": "rows_length_ge_4", "value": count_ge4},
        {"metric": "unique_sequences_ge_4", "value": unique_ge4},
        {"metric": "rows_with_internal_repeats", "value": int(total_repeated_rows)},
        {"metric": "out_of_range_date_rows", "value": len(junk_dates_df)},
        {"metric": "unique_techniques_uncollapsed", "value": unique_fine_grained},
        {"metric": "unique_techniques_collapsed", "value": unique_parents},
    ]
    df_eda_summary = pd.DataFrame(eda_summary_records)
    summary_csv_path = results_dir / "eda_summary.csv"
    df_eda_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved EDA summary table to: {summary_csv_path}")
    
    # 9. Publication-Quality Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Panel 1: Sequence Length Distribution (Log Scale)
    ax1 = axes[0, 0]
    lengths_plot = length_counts[length_counts.index <= 15]
    ax1.bar(lengths_plot.index, lengths_plot.values, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_xlabel("Sequence Length", fontweight="bold")
    ax1.set_ylabel("Count (Log Scale)", fontweight="bold")
    ax1.set_title("A. Sequence Length Distribution (All 1.04M Rows)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Highlight singletons vs length >= 4
    ax1.annotate(f"Singletons\n{singleton_pct:.1f}%",
                 xy=(1, lengths_plot[1]), xytext=(2.5, lengths_plot[1] * 0.4),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                 fontweight="bold", color="#d62728")
    
    # Panel 2: Top 15 Techniques Overall (Global Priors)
    ax2 = axes[0, 1]
    top15_global = global_tech_counts.head(15).sort_values(ascending=True)
    ax2.barh(top15_global.index, top15_global.values, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Total Observations", fontweight="bold")
    ax2.set_title("B. Top 15 Techniques (Global Unigram Prior)", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.6)
    
    # Panel 3: Top 15 Techniques in Multi-Stage Sequences (Length >= 4)
    ax3 = axes[1, 0]
    top15_seq = seq_tech_counts.head(15).sort_values(ascending=True)
    ax3.barh(top15_seq.index, top15_seq.values, color="#ff7f0e", edgecolor="black", alpha=0.85)
    ax3.set_xlabel("Occurrences in Length >= 4 Sequences", fontweight="bold")
    ax3.set_title("C. Top 15 Techniques in Multi-Stage Sequences", fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.6)
    
    # Panel 4: Monthly Sighting Volume Timeline (Valid Dates Only)
    ax4 = axes[1, 1]
    df_valid_dates = df_raw[valid_year_mask].copy()
    monthly_counts = df_valid_dates.set_index("date_parsed").resample("ME").size()
    ax4.plot(monthly_counts.index, monthly_counts.values, marker="o", color="#9467bd", linewidth=2)
    ax4.set_xlabel("Sighting Date", fontweight="bold")
    ax4.set_ylabel("Monthly Sighting Volume", fontweight="bold")
    ax4.set_title("D. Sighting Volume Timeline (Clean Dates)", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=30)
    
    plot_path = results_dir / "eda_overview.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved EDA visualization to: {plot_path}")
    
    return {
        "total_rows": total_raw_rows,
        "singleton_pct": singleton_pct,
        "count_ge4": count_ge4,
        "unique_ge4": unique_ge4,
        "repeats": total_repeated_rows,
        "summary_csv": summary_csv_path,
        "priors_csv": priors_csv_path,
        "plot_path": plot_path
    }


if __name__ == "__main__":
    run_eda(config.SIGHTINGS_DATA_PATH, config.RESULTS_DIR)
