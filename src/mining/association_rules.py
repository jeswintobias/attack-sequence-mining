"""
Task 2 – FP-Growth & Apriori Association Rule Mining on Train Itemsets.

Uses mlxtend's fpgrowth and apriori on train sequences treated as
unordered itemsets (set of techniques). We mine frequent itemsets (max_len=4)
at three minimum support levels, then extract association rules with
confidence >= 0.5. We report rule counts, top-30 by lift, and runtime
comparison (Apriori vs FP-Growth) as a table and line chart.

Support thresholds justification:
  - 0.01 (1%): Captures broadly recurring patterns — roughly ≥16 unique
    sequences out of 1,633.
  - 0.03 (3%): Moderate threshold — roughly ≥49 unique sequences.
  - 0.05 (5%): Conservative — roughly ≥82 unique sequences.

Design choices (viva):
  - Itemsets are UNORDERED: FP-Growth mines co-occurrence, not sequence order.
    Sequential patterns are handled separately by PrefixSpan.
  - Training data only, weighted by dup_count (each unique sequence is
    replicated dup_count times in the transaction database).
  - Confidence ≥ 0.5 filters spurious rules.
  - Runtime comparison uses wall-clock time over 10 runs (mean and std).
    No scaling claim is made from this small transaction count.
  - A unique-sequence sensitivity run counts each train sequence once (N=1,633).

References:
  - FP-Growth algorithm: Han, Pei & Yin (2000).
  - mlxtend implementation: Raschka (2018).
"""

import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import config


def _load_train_itemsets() -> tuple:
    """Return (unique itemsets list, dup_count list) from train sequences."""
    train_df = pd.read_csv(
        config.BASE_DIR / "data" / "processed" / "train_sequences.csv"
    )
    itemsets = []
    weights = []
    for _, r in train_df.iterrows():
        seq = ast.literal_eval(r["sequence"])
        itemsets.append(list(set(seq)))
        weights.append(int(r["dup_count"]))
    return itemsets, weights


def build_train_transactions() -> list:
    """
    Build a list of itemsets from train sequences,
    each replicated dup_count times for duplicate-weighted mining.
    """
    itemsets, weights = _load_train_itemsets()
    transactions = []
    for itemset, w in zip(itemsets, weights):
        for _ in range(w):
            transactions.append(itemset)
    print(f"Built {len(transactions):,} transactions "
          f"({len(itemsets)} unique sequences × dup_count)")
    return transactions


def build_train_transactions_unique() -> list:
    """Build one transaction per unique train sequence (N unique sequences)."""
    itemsets, _ = _load_train_itemsets()
    print(f"Built {len(itemsets):,} unique-sequence transactions")
    return itemsets


def encode_transactions(transactions: list) -> pd.DataFrame:
    """One-hot encode the transaction list for mlxtend."""
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df


def run_association_mining(results_dir: Path = None):
    """Mine frequent itemsets and association rules at three support levels."""
    if results_dir is None:
        results_dir = config.ASSOCIATION_RESULTS_DIR
    fig_dir = config.EDA_RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Build transactions
    transactions = build_train_transactions()
    df_encoded = encode_transactions(transactions)
    n_transactions = len(df_encoded)
    print(f"Encoded {n_transactions:,} transactions, "
          f"{len(df_encoded.columns)} unique items")

    # Support thresholds
    supports = [0.01, 0.03, 0.05]
    min_confidence = 0.5
    max_len = 4
    n_timing_runs = 10

    # ── Mine at each support and collect results ────────────────────────────
    all_rules_summaries = []
    timing_records = []
    top30_frames = []

    for min_sup in supports:
        print(f"\n{'='*60}")
        print(f"Support = {min_sup} ({min_sup*100:.0f}%)")
        print(f"{'='*60}")

        # FP-Growth timed
        fp_times = []
        for run in range(n_timing_runs):
            t0 = time.perf_counter()
            fi_fp = fpgrowth(df_encoded, min_support=min_sup,
                             use_colnames=True, max_len=max_len)
            fp_times.append(time.perf_counter() - t0)
        fp_mean = float(np.mean(fp_times))
        fp_std = float(np.std(fp_times, ddof=1))
        print(f"  FP-Growth: {len(fi_fp)} frequent itemsets, "
              f"mean time = {fp_mean:.3f}s (std {fp_std:.4f}s)")

        # Apriori timed
        ap_times = []
        for run in range(n_timing_runs):
            t0 = time.perf_counter()
            fi_ap = apriori(df_encoded, min_support=min_sup,
                            use_colnames=True, max_len=max_len)
            ap_times.append(time.perf_counter() - t0)
        ap_mean = float(np.mean(ap_times))
        ap_std = float(np.std(ap_times, ddof=1))
        print(f"  Apriori:   {len(fi_ap)} frequent itemsets, "
              f"mean time = {ap_mean:.3f}s (std {ap_std:.4f}s)")

        # Verify same itemsets
        assert len(fi_fp) == len(fi_ap), (
            f"Mismatch: FP-Growth={len(fi_fp)}, Apriori={len(fi_ap)}"
        )

        timing_records.append({
            "min_support": min_sup,
            "n_frequent_itemsets": len(fi_fp),
            "n_timing_runs": n_timing_runs,
            "fpgrowth_time_mean_s": round(fp_mean, 4),
            "fpgrowth_time_std_s": round(fp_std, 4),
            "apriori_time_mean_s": round(ap_mean, 4),
            "apriori_time_std_s": round(ap_std, 4),
            "speedup_mean": round(ap_mean / fp_mean, 2) if fp_mean > 0 else float("inf"),
        })

        # Extract association rules
        if len(fi_fp) > 0:
            rules = association_rules(
                fi_fp, metric="confidence", min_threshold=min_confidence,
                num_itemsets=n_transactions
            )
            # Clean up frozenset display
            rules["antecedents_str"] = rules["antecedents"].apply(
                lambda x: ", ".join(sorted(x))
            )
            rules["consequents_str"] = rules["consequents"].apply(
                lambda x: ", ".join(sorted(x))
            )
            n_rules = len(rules)
            print(f"  Rules (conf >= {min_confidence}): {n_rules}")
        else:
            rules = pd.DataFrame()
            n_rules = 0

        all_rules_summaries.append({
            "min_support": min_sup,
            "n_frequent_itemsets": len(fi_fp),
            "n_rules_conf_ge_05": n_rules,
        })

        # Top 30 by lift
        if n_rules > 0:
            top30 = rules.nlargest(30, "lift").copy()
            top30["min_support_threshold"] = min_sup
            top30_frames.append(top30)
            print(f"  Top-30 rules by lift (lift range: "
                  f"{top30['lift'].min():.2f} – {top30['lift'].max():.2f})")

    # ── Save summary tables ─────────────────────────────────────────────────
    df_summary = pd.DataFrame(all_rules_summaries)
    df_summary.to_csv(results_dir / "fpgrowth_rule_counts.csv", index=False)

    df_timing = pd.DataFrame(timing_records)
    df_timing.to_csv(results_dir / "fpgrowth_vs_apriori_timing.csv", index=False)

    # Concatenate top-30 rules from all support levels
    if top30_frames:
        df_top30 = pd.concat(top30_frames, ignore_index=True)
        cols = ["min_support_threshold", "antecedents_str", "consequents_str",
                "support", "confidence", "lift", "leverage", "conviction"]
        available_cols = [c for c in cols if c in df_top30.columns]
        df_top30[available_cols].to_csv(
            results_dir / "fpgrowth_top30_by_lift.csv", index=False
        )

    print(f"\n── Rule Count Summary ──")
    print(df_summary.to_string(index=False))
    print(f"\n── Runtime Comparison ──")
    print(df_timing.to_string(index=False))

    # ── Runtime comparison line chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [str(s) for s in df_timing["min_support"]],
        df_timing["apriori_time_mean_s"],
        marker="o", linewidth=2, color="#e15759", label="Apriori"
    )
    ax.plot(
        [str(s) for s in df_timing["min_support"]],
        df_timing["fpgrowth_time_mean_s"],
        marker="s", linewidth=2, color="#4e79a7", label="FP-Growth"
    )
    ax.set_xlabel("Minimum Support")
    ax.set_ylabel("Mean runtime over 10 runs (seconds)")
    ax.set_title("FP-Growth vs Apriori Runtime Comparison\n"
                 f"(max_len={max_len}, conf ≥ {min_confidence}, "
                 f"N={n_transactions:,} transactions)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "16_fpgrowth_vs_apriori_runtime.png", dpi=300)
    plt.close(fig)
    print(f"\nSaved 16_fpgrowth_vs_apriori_runtime.png")

    unique_out = run_unique_fpgrowth(results_dir=results_dir)

    return {
        "df_summary": df_summary,
        "df_timing": df_timing,
        "df_unique_summary": unique_out["df_summary"],
    }


def _rules_from_itemsets(fi, n_transactions: int, min_confidence: float) -> pd.DataFrame:
    if len(fi) == 0:
        return pd.DataFrame()
    rules = association_rules(
        fi, metric="confidence", min_threshold=min_confidence,
        num_itemsets=n_transactions,
    )
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    return rules


def run_unique_fpgrowth(results_dir: Path = None):
    """
    FP-Growth sensitivity: each unique train sequence counted once (N = 1,633).
    Supports 1%, 3%, 5% of N. Saves rule counts and top-10 rules by lift.
    """
    if results_dir is None:
        results_dir = config.ASSOCIATION_RESULTS_DIR

    transactions = build_train_transactions_unique()
    df_encoded = encode_transactions(transactions)
    n_transactions = len(df_encoded)
    print(f"\nUnique-sequence FP-Growth: N = {n_transactions:,}")

    supports = [0.01, 0.03, 0.05]
    min_confidence = 0.5
    max_len = 4
    summaries = []
    top10_frames = []

    for min_sup in supports:
        fi = fpgrowth(df_encoded, min_support=min_sup,
                      use_colnames=True, max_len=max_len)
        rules = _rules_from_itemsets(fi, n_transactions, min_confidence)
        n_rules = len(rules)
        summaries.append({
            "min_support": min_sup,
            "n_transactions": n_transactions,
            "n_frequent_itemsets": len(fi),
            "n_rules_conf_ge_05": n_rules,
        })
        print(f"  unique sup={min_sup}: itemsets={len(fi)}, rules={n_rules}")
        if n_rules > 0:
            top10 = rules.nlargest(10, "lift").copy()
            top10["min_support_threshold"] = min_sup
            top10_frames.append(top10)
            out_cols = [
                "min_support_threshold", "antecedents_str", "consequents_str",
                "support", "confidence", "lift",
            ]
            top10[out_cols].to_csv(
                results_dir / f"fpgrowth_unique_top10_sup{int(min_sup*100):02d}.csv",
                index=False,
            )

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(results_dir / "fpgrowth_unique_rule_counts.csv", index=False)

    if top10_frames:
        df_top = pd.concat(top10_frames, ignore_index=True)
        cols = [
            "min_support_threshold", "antecedents_str", "consequents_str",
            "support", "confidence", "lift",
        ]
        df_top[cols].to_csv(
            results_dir / "fpgrowth_unique_top10_by_lift.csv", index=False
        )

    return {"df_summary": df_summary}


if __name__ == "__main__":
    run_association_mining()
