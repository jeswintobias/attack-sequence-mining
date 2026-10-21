"""
Data Preprocessing and Prefix-Disjoint Splitting (Stage A).

Design Decisions (for Viva Defense):
1. Safe Parsing: Safely parses stringified Python lists using ast.literal_eval.
   Validates timestamps (%m/%d/%y), flagging out-of-range years (<2000 or >2026).
2. Pure In-Memory Transformations: Raw dataset files remain completely immutable on disk.
   All filtering (singletons, repetitions, collapsing) happens in pandas DataFrames.
3. Deduplication of Repetitions: If an attacker repeats the same technique within a sighting,
   we retain only the first occurrence to model transitions between distinct tactical states.
4. Subtechnique Collapsing: Optional aggregation of granular sub-techniques to parent techniques
   (e.g., T1055.012 -> T1055) to assess model generalization at the broader tactical level.
   Evaluated for both settings (True and False).
5. Length Threshold (>= 4): Sequences with length < 4 are filtered out because 2nd-order Markov
   and sequential models require at least prefix length 2 plus subsequent prediction targets.
6. Prefix-Disjoint Splitting: Rather than random example splitting (which causes data leakage by placing
   identical prefixes in both train and test sets), we split unique prefixes strictly 80/20 so the test
   evaluation measures generalization to unseen contexts.
"""

import ast
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import config


def collapse_technique(tech_id: str) -> str:
    """Collapses a sub-technique ID (e.g., 'T1055.012') to parent ID ('T1055')."""
    return tech_id.split(".")[0]


def deduplicate_consecutive(sequence: List[str]) -> List[str]:
    """
    Keeps only the first occurrence when a technique repeats inside a list.
    Viva justification: Eliminates self-loops so models learn transitions between distinct attack actions.
    """
    deduped = []
    for item in sequence:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped


def deduplicate_first_occurrence(sequence: List[str]) -> List[str]:
    """
    Preserves order of first appearance for each unique technique in the list.
    Handles intra-list repetitions (e.g., [T1, T2, T1] -> [T1, T2]).
    """
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_and_audit_sightings(filepath: Path) -> pd.DataFrame:
    """
    Loads sightings_v2_public.csv in-memory.
    Parses tid safely with ast.literal_eval and validates dates (%m/%d/%y).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Sightings file not found at: {filepath}")
        
    print(f"Loading sightings dataset from {filepath} (raw file remains immutable)...")
    df = pd.read_csv(filepath)
    
    # 1. Parse tid safely with ast.literal_eval
    df["tid_parsed"] = df["tid"].apply(ast.literal_eval)
    df["raw_length"] = df["tid_parsed"].apply(len)
    
    # 2. Parse dates with %m/%d/%y and flag out-of-range years
    df["date_parsed"] = pd.to_datetime(df["sighting_date"], format="%m/%d/%y", errors="coerce")
    out_of_range = df[
        (df["date_parsed"].dt.year < 2000) | (df["date_parsed"].dt.year > 2026)
    ]
    print(f"Loaded {len(df):,} rows. Flagged {len(out_of_range)} sightings with out-of-range years (<2000 or >2026).")
    return df


def extract_sequences_from_sightings(
    df: pd.DataFrame,
    collapse_subtechniques: bool = False,
    min_length: int = 4
) -> Tuple[List[Tuple[str, ...]], pd.Series]:
    """
    Extracts sequences of length >= min_length from parsed sightings DataFrame.
    Applies deduplication of internal repetitions and optional subtechnique collapsing.
    
    Returns:
        all_sequences: List of all extracted sequences of length >= min_length.
        sequence_counts: Value counts of unique deduplicated sequences.
    """
    processed_sequences = []
    
    for raw_techs in df["tid_parsed"]:
        # 1. Subtechnique collapsing if enabled
        if collapse_subtechniques:
            techs = [collapse_technique(t) for t in raw_techs]
        else:
            techs = list(raw_techs)
            
        # 2. Intra-list repetition deduplication: keep first occurrence
        cleaned_techs = deduplicate_first_occurrence(techs)
        
        # 3. Filter sequences of length >= min_length
        if len(cleaned_techs) >= min_length:
            processed_sequences.append(tuple(cleaned_techs))
            
    sequence_counts = pd.Series(processed_sequences).value_counts()
    return processed_sequences, sequence_counts


def generate_prefix_examples(sequences: List[Tuple[str, ...]]) -> pd.DataFrame:
    """
    Builds (prefix, next) training/testing examples for every prefix length from 2 up to n-1.
    
    For a sequence (T1, T2, T3, T4):
    - prefix: (T1, T2) -> next: T3
    - prefix: (T1, T2, T3) -> next: T4
    """
    examples = []
    for seq in sequences:
        n = len(seq)
        for k in range(2, n):
            prefix = seq[:k]
            next_token = seq[k]
            examples.append({
                "prefix": prefix,
                "next": next_token,
                "prefix_len": k,
                "full_seq": seq
            })
    return pd.DataFrame(examples)


def prefix_disjoint_split(
    df_examples: pd.DataFrame,
    split_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[Tuple[str, ...]], Set[Tuple[str, ...]]]:
    """
    Performs prefix-disjoint split:
    Every unique prefix is assigned strictly to train or strictly to test.
    
    Includes an explicit assertion to mathematically guarantee zero prefix overlap.
    """
    unique_prefixes = df_examples["prefix"].unique()
    rng = np.random.RandomState(random_seed)
    shuffled_prefixes = rng.permutation(unique_prefixes)
    
    split_idx = int(split_ratio * len(shuffled_prefixes))
    train_prefixes = set(shuffled_prefixes[:split_idx])
    test_prefixes = set(shuffled_prefixes[split_idx:])
    
    # Strict disjointness assertion
    overlap = train_prefixes.intersection(test_prefixes)
    assert len(overlap) == 0, f"Critical error: Prefix overlap detected! Overlap: {overlap}"
    
    df_train = df_examples[df_examples["prefix"].isin(train_prefixes)].copy().reset_index(drop=True)
    df_test = df_examples[df_examples["prefix"].isin(test_prefixes)].copy().reset_index(drop=True)
    
    return df_train, df_test, train_prefixes, test_prefixes


def run_stage_a(data_path: Path, results_dir: Path) -> Dict[str, Any]:
    """
    Executes Stage A for both collapse_subtechniques=False and collapse_subtechniques=True.
    Saves results/data_stats.csv and results/length_histogram.png.
    """
    df_raw = load_and_audit_sightings(data_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    stats_records = []
    length_distributions = {}
    processed_datasets = {}
    
    for collapse in [False, True]:
        setting_name = "Collapsed" if collapse else "Uncollapsed"
        print(f"\n--- Processing Setting: {setting_name} (collapse_subtechniques={collapse}) ---")
        
        all_seqs, seq_counts = extract_sequences_from_sightings(
            df_raw,
            collapse_subtechniques=collapse,
            min_length=config.MIN_SEQ_LEN
        )
        
        unique_seqs = list(seq_counts.index)
        print(f"Total sequences (length >= {config.MIN_SEQ_LEN}): {len(all_seqs):,}")
        print(f"Unique deduplicated sequences: {len(unique_seqs):,}")
        
        # Build prefix examples from unique sequences
        df_examples = generate_prefix_examples(unique_seqs)
        print(f"Generated (prefix, next) examples: {len(df_examples):,}")
        
        # Prefix-disjoint split
        df_train, df_test, train_pref, test_pref = prefix_disjoint_split(
            df_examples,
            split_ratio=config.TRAIN_TEST_SPLIT_RATIO,
            random_seed=config.RANDOM_SEED
        )
        print(f"Unique prefixes: {len(df_examples['prefix'].unique()):,} -> Train: {len(train_pref):,}, Test: {len(test_pref):,}")
        print(f"Split examples -> Train: {len(df_train):,}, Test: {len(df_test):,}")
        
        # Technique occurrences across multi-stage sequences
        all_techs = [tech for seq in unique_seqs for tech in seq]
        top_techs = pd.Series(all_techs).value_counts().head(5).to_dict()
        
        # Length distribution of unique sequences
        len_dist = pd.Series([len(s) for s in unique_seqs]).value_counts().sort_index().to_dict()
        length_distributions[setting_name] = len_dist
        
        stats_records.append({
            "setting": setting_name,
            "collapse_subtechniques": collapse,
            "total_extracted_sequences": len(all_seqs),
            "unique_sequences": len(unique_seqs),
            "min_sequence_length": min(len(s) for s in unique_seqs),
            "max_sequence_length": max(len(s) for s in unique_seqs),
            "total_prefix_examples": len(df_examples),
            "unique_prefixes": len(df_examples["prefix"].unique()),
            "train_unique_prefixes": len(train_pref),
            "test_unique_prefixes": len(test_pref),
            "train_examples": len(df_train),
            "test_examples": len(df_test),
            "top_5_techniques": str(top_techs)
        })
        
        processed_datasets[setting_name] = {
            "df_examples": df_examples,
            "df_train": df_train,
            "df_test": df_test,
            "unique_seqs": unique_seqs,
            "seq_counts": seq_counts
        }
        
    df_stats = pd.DataFrame(stats_records)
    stats_csv_path = results_dir / "data_stats.csv"
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"\nSaved dataset statistics to: {stats_csv_path}")
    
    # Generate and save sequence length histogram figure
    fig, ax = plt.subplots(figsize=(10, 5))
    all_lengths = sorted(list(set(
        list(length_distributions["Uncollapsed"].keys()) +
        list(length_distributions["Collapsed"].keys())
    )))
    
    x = np.arange(len(all_lengths))
    width = 0.38
    
    y_uncoll = [length_distributions["Uncollapsed"].get(l, 0) for l in all_lengths]
    y_coll = [length_distributions["Collapsed"].get(l, 0) for l in all_lengths]
    
    rects1 = ax.bar(x - width/2, y_uncoll, width, label="Uncollapsed (Fine-grained)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, y_coll, width, label="Collapsed (Parent)", color="#d95f02")
    
    ax.set_xlabel("Attack Sequence Length", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Unique Sequences", fontsize=11, fontweight="bold")
    ax.set_title("Distribution of Unique Sequence Lengths (Sightings Dataset)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in all_lengths])
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f"{height}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=45)
            
    plt.tight_layout()
    hist_fig_path = results_dir / "length_histogram.png"
    plt.savefig(hist_fig_path, dpi=300)
    plt.close()
    print(f"Saved sequence length histogram figure to: {hist_fig_path}")
    
    return {
        "df_stats": df_stats,
        "processed_datasets": processed_datasets,
        "stats_csv_path": stats_csv_path,
        "hist_fig_path": hist_fig_path
    }


if __name__ == "__main__":
    run_stage_a(config.DATA_PATH, config.EDA_RESULTS_DIR)
