"""
Stage A – Sequence-Level 60/20/20 Stratified Split.

Design decisions (for viva defense):
1. Deduplication rule: The existing code uses deduplicate_first_occurrence(), which
   keeps only the first occurrence of each technique in a list. For example,
   [T1, T2, T1, T3] → [T1, T2, T3]. This removes ALL duplicates, not just
   consecutive ones. We document and reuse this rule. (MEASURED)
2. Whole-sequence split: We split at the level of unique sequences (not prefixes
   or examples) to guarantee that no full sequence appears in more than one set.
3. Stratification by length bucket: Ensures train/val/test have similar length
   distributions. Buckets are {4, 5, 6, 7+}.
4. Duplicate counts: Each unique sequence carries its row count from the original
   CSV. This allows duplicate-weighted evaluation later.
5. Prefix/next examples are generated AFTER splitting, so there is zero leakage
   of sequence identity across sets.
"""

import ast
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import config


# ── helpers (reused from preprocessing.py) ────────────────────────────────────

def collapse_technique(tech_id: str) -> str:
    """Collapse sub-technique ID to parent (e.g. T1055.012 → T1055)."""
    return tech_id.split(".")[0]


def deduplicate_first_occurrence(sequence: List[str]) -> List[str]:
    """
    Keep only the first occurrence of each technique.
    Example: [T1, T2, T1, T3] → [T1, T2, T3].
    This is the rule used throughout the project (not just consecutive dedup).
    """
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ── core pipeline ─────────────────────────────────────────────────────────────

def load_and_extract(
    data_path: Path,
    collapse_subtechniques: bool = True,
    min_length: int = 4,
) -> pd.DataFrame:
    """
    Load raw CSV, collapse subtechniques, deduplicate first-occurrence,
    filter length >= min_length, and return a DataFrame of unique sequences
    with their duplicate counts.

    Returns DataFrame with columns:
        seq_tuple  – tuple of technique IDs
        seq_len    – length of the sequence
        dup_count  – number of raw rows mapping to this unique sequence
    """
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} raw rows from {data_path.name}")

    sequences = []
    for raw in df["tid"].apply(ast.literal_eval):
        if collapse_subtechniques:
            techs = [collapse_technique(t) for t in raw]
        else:
            techs = list(raw)
        cleaned = deduplicate_first_occurrence(techs)
        if len(cleaned) >= min_length:
            sequences.append(tuple(cleaned))

    seq_series = pd.Series(sequences)
    counts = seq_series.value_counts().reset_index()
    counts.columns = ["seq_tuple", "dup_count"]
    counts["seq_len"] = counts["seq_tuple"].apply(len)
    print(f"Extracted {len(seq_series):,} rows with length >= {min_length}")
    print(f"  → {len(counts):,} unique sequences")
    return counts


def length_bucket(length: int) -> str:
    """Assign a sequence length to its stratification bucket."""
    if length <= 6:
        return str(length)
    return "7+"


def stratified_split(
    df_unique: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split unique sequences 60/20/20 stratified by length bucket.
    Each row carries its dup_count so duplicates are preserved.
    """
    df = df_unique.copy()
    df["bucket"] = df["seq_len"].apply(length_bucket)

    train_parts, val_parts, test_parts = [], [], []
    rng = np.random.RandomState(seed)

    for bucket, group in df.groupby("bucket"):
        indices = group.index.values.copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Remainder goes to test
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
        test_parts.append(df.loc[test_idx])

    df_train = pd.concat(train_parts).reset_index(drop=True)
    df_val = pd.concat(val_parts).reset_index(drop=True)
    df_test = pd.concat(test_parts).reset_index(drop=True)
    return df_train, df_val, df_test


def generate_examples(df_seqs: pd.DataFrame) -> pd.DataFrame:
    """
    From a set of unique sequences (with dup_count), generate
    (prefix, next) examples for prefix lengths 2 up to n-1.
    """
    rows = []
    for _, r in df_seqs.iterrows():
        seq = r["seq_tuple"]
        dup = r["dup_count"]
        n = len(seq)
        for k in range(2, n):
            rows.append({
                "prefix": seq[:k],
                "next": seq[k],
                "prefix_len": k,
                "full_seq": seq,
                "dup_count": dup,
            })
    return pd.DataFrame(rows)


def consecutive_pairs_in_train(df_eval: pd.DataFrame, train_seqs: list) -> float:
    """
    Fraction of (last_prefix_item, next_item) pairs in the evaluation set
    that appear as consecutive items somewhere in a train sequence.
    """
    # Build set of all consecutive (A, B) bigrams in train
    train_bigrams = set()
    for seq in train_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            train_bigrams.add((a, b))

    if len(df_eval) == 0:
        return 0.0

    seen = 0
    for _, r in df_eval.iterrows():
        last_item = r["prefix"][-1]
        next_item = r["next"]
        if (last_item, next_item) in train_bigrams:
            seen += 1
    return seen / len(df_eval)


def save_set(df_seqs: pd.DataFrame, path: Path):
    """Save a sequence set to CSV with readable columns."""
    out = df_seqs.copy()
    out["sequence"] = out["seq_tuple"].apply(lambda t: str(list(t)))
    out[["sequence", "seq_len", "dup_count"]].to_csv(path, index=False)
    print(f"  Saved {path.name} ({len(out)} unique seqs)")


def run_split(data_path: Path = None, results_dir: Path = None):
    """Execute the full split pipeline and save all outputs."""
    if data_path is None:
        data_path = config.SIGHTINGS_DATA_PATH
    if results_dir is None:
        results_dir = config.EDA_RESULTS_DIR

    processed_dir = config.BASE_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract unique sequences (collapsed, first-occurrence dedup) ──
    df_unique = load_and_extract(
        data_path,
        collapse_subtechniques=config.COLLAPSE_SUBTECHNIQUES,
        min_length=config.MIN_SEQ_LEN,
    )

    # ── Step 2: Stratified 60/20/20 split ─────────────────────────────────────
    df_train, df_val, df_test = stratified_split(
        df_unique,
        train_frac=0.60,
        val_frac=0.20,
        seed=config.RANDOM_SEED,
    )

    # Save each set
    save_set(df_train, processed_dir / "train_sequences.csv")
    save_set(df_val, processed_dir / "val_sequences.csv")
    save_set(df_test, processed_dir / "test_sequences.csv")

    # ── Step 3: Assert no sequence in more than one set ───────────────────────
    train_set = set(df_train["seq_tuple"])
    val_set = set(df_val["seq_tuple"])
    test_set = set(df_test["seq_tuple"])

    assert len(train_set & val_set) == 0, "FAIL: train ∩ val overlap"
    assert len(train_set & test_set) == 0, "FAIL: train ∩ test overlap"
    assert len(val_set & test_set) == 0, "FAIL: val ∩ test overlap"
    assert len(train_set) + len(val_set) + len(test_set) == len(df_unique), \
        "FAIL: sets don't partition all unique sequences"
    print("\n✓ PASS: No full sequence appears in more than one set")
    print("✓ PASS: Sets partition all unique sequences exactly")

    # ── Step 4: Generate (prefix, next) examples AFTER splitting ──────────────
    ex_train = generate_examples(df_train)
    ex_val = generate_examples(df_val)
    ex_test = generate_examples(df_test)

    ex_train.to_pickle(processed_dir / "train_examples.pkl")
    ex_val.to_pickle(processed_dir / "val_examples.pkl")
    ex_test.to_pickle(processed_dir / "test_examples.pkl")

    # ── Step 5: Compute leakage metric ────────────────────────────────────────
    train_seqs_list = df_train["seq_tuple"].tolist()
    val_pair_share = consecutive_pairs_in_train(ex_val, train_seqs_list)
    test_pair_share = consecutive_pairs_in_train(ex_test, train_seqs_list)
    print(f"\nShare of val (last, next) pairs seen in train bigrams:  {val_pair_share:.4f}")
    print(f"Share of test (last, next) pairs seen in train bigrams: {test_pair_share:.4f}")

    # ── Step 6: Report sizes and bucket mix ───────────────────────────────────
    def bucket_mix(df_s):
        buckets = df_s["seq_len"].apply(length_bucket).value_counts().sort_index()
        return buckets.to_dict()

    stats_rows = []
    for name, df_s, df_e in [
        ("train", df_train, ex_train),
        ("val", df_val, ex_val),
        ("test", df_test, ex_test),
    ]:
        bm = bucket_mix(df_s)
        stats_rows.append({
            "set": name,
            "unique_sequences": len(df_s),
            "dup_weighted_sequences": int(df_s["dup_count"].sum()),
            "total_examples": len(df_e),
            "dup_weighted_examples": int(df_e["dup_count"].sum()),
            "bucket_4": bm.get("4", 0),
            "bucket_5": bm.get("5", 0),
            "bucket_6": bm.get("6", 0),
            "bucket_7+": bm.get("7+", 0),
        })

    df_stats = pd.DataFrame(stats_rows)
    # Add leakage rows
    df_leakage = pd.DataFrame([{
        "metric": "val_pair_share_in_train",
        "value": round(val_pair_share, 6),
    }, {
        "metric": "test_pair_share_in_train",
        "value": round(test_pair_share, 6),
    }])

    stats_path = results_dir / "data_stats.csv"
    df_stats.to_csv(stats_path, index=False)
    df_leakage.to_csv(results_dir / "leakage_stats.csv", index=False)

    print(f"\nSaved data_stats.csv and leakage_stats.csv")
    print("\n── Set sizes ──")
    print(df_stats.to_string(index=False))
    print("\n── Leakage ──")
    print(df_leakage.to_string(index=False))

    return {
        "df_train_seqs": df_train,
        "df_val_seqs": df_val,
        "df_test_seqs": df_test,
        "ex_train": ex_train,
        "ex_val": ex_val,
        "ex_test": ex_test,
        "val_pair_share": val_pair_share,
        "test_pair_share": test_pair_share,
        "df_stats": df_stats,
    }


if __name__ == "__main__":
    run_split()
