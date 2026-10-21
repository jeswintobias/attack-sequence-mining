"""
Test suite for our custom PrefixSpan implementation.

Validates exact match of patterns and supports against the pip 'prefixspan'
package on:
  1. A small hand-crafted sample database.
  2. The full train sequence database.

The pip package is imported ONLY here in tests; production code uses
src/prefixspan.py exclusively.
"""

import ast
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"

# Ensure we import the PIP package from site-packages, not src/mining/prefixspan.py
_saved_sys_path = list(sys.path)
sys.path = [p for p in sys.path if p != '' and not p.endswith('/src') and not p.endswith('/mining') and p != str(Path(__file__).resolve().parent)]
import prefixspan as pip_prefixspan_pkg
sys.path = _saved_sys_path

# Allow imports from src/ and base
for p in [str(ROOT), str(SRC_DIR), str(SRC_DIR / "core"), str(SRC_DIR / "mining")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import OUR implementation explicitly by path
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "our_prefixspan",
    SRC_DIR / "mining" / "prefixspan.py"
)
our_ps_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(our_ps_module)
prefixspan_mine = our_ps_module.prefixspan_mine


def _pip_mine(sequences, min_support, max_len):
    """
    Mine using pip prefixspan package. Returns list of (support, pattern)
    sorted by support desc, then pattern.
    """
    ps = pip_prefixspan_pkg.PrefixSpan(sequences)
    ps.maxlen = max_len
    raw = ps.frequent(min_support)
    # Filter out empty patterns
    results = [(s, p) for s, p in raw if len(p) > 0]
    results.sort(key=lambda x: (-x[0], x[1]))
    return results


def test_sample_database():
    """Test on a small hand-crafted database with known expected results."""
    print("=" * 60)
    print("TEST 1: Small sample database")
    print("=" * 60)

    db = [
        ('a', 'b', 'c'),
        ('a', 'b', 'd'),
        ('a', 'c', 'd'),
        ('b', 'c', 'd'),
        ('a', 'b', 'c', 'd'),
    ]
    min_sup = 2
    max_len = 3

    # Our implementation
    ours = prefixspan_mine(db, min_support=min_sup, max_len=max_len)
    # Pip implementation
    pip_results = _pip_mine([list(s) for s in db], min_support=min_sup, max_len=max_len)

    print(f"  Our implementation: {len(ours)} patterns")
    print(f"  Pip implementation: {len(pip_results)} patterns")

    # Convert to comparable format
    ours_set = {(s, tuple(p)) for s, p in ours}
    pip_set = {(s, tuple(p)) for s, p in pip_results}

    if ours_set == pip_set:
        print("  ✓ EXACT MATCH on sample database")
    else:
        only_ours = ours_set - pip_set
        only_pip = pip_set - ours_set
        print(f"  ✗ MISMATCH!")
        if only_ours:
            print(f"    Only in ours: {sorted(only_ours)}")
        if only_pip:
            print(f"    Only in pip:  {sorted(only_pip)}")
        raise AssertionError("Sample database test FAILED")

    return True


def test_full_train():
    """Test on the full train sequence database."""
    print("\n" + "=" * 60)
    print("TEST 2: Full train database")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent.parent
    train_df = pd.read_csv(base_dir / "data" / "processed" / "train_sequences.csv")

    sequences = []
    weights = []
    for _, r in train_df.iterrows():
        seq = tuple(ast.literal_eval(r["sequence"]))
        sequences.append(seq)
        weights.append(r["dup_count"])

    total_weighted = sum(weights)
    # Use 5% support
    min_sup_abs = int(total_weighted * 0.05)
    max_len = 4
    print(f"  Sequences: {len(sequences)} unique, {total_weighted} weighted")
    print(f"  Min support: {min_sup_abs} (5% of {total_weighted})")
    print(f"  Max len: {max_len}")

    # Our implementation (weighted)
    ours = prefixspan_mine(sequences, min_support=min_sup_abs,
                           max_len=max_len, weights=weights)

    # Pip implementation: expand sequences by dup_count
    expanded = []
    for seq, w in zip(sequences, weights):
        for _ in range(w):
            expanded.append(list(seq))
    pip_results = _pip_mine(expanded, min_support=min_sup_abs, max_len=max_len)

    print(f"  Our implementation: {len(ours)} patterns")
    print(f"  Pip implementation: {len(pip_results)} patterns")

    ours_set = {(s, tuple(p)) for s, p in ours}
    pip_set = {(s, tuple(p)) for s, p in pip_results}

    if ours_set == pip_set:
        print("  ✓ EXACT MATCH on full train database")
    else:
        only_ours = ours_set - pip_set
        only_pip = pip_set - ours_set
        print(f"  ✗ MISMATCH!")
        print(f"    Only in ours ({len(only_ours)}): {sorted(list(only_ours))[:5]}")
        print(f"    Only in pip  ({len(only_pip)}): {sorted(list(only_pip))[:5]}")
        raise AssertionError("Full train database test FAILED")

    # Print top-10 patterns
    print(f"\n  Top-10 patterns by support:")
    for i, (sup, pat) in enumerate(ours[:10]):
        print(f"    {i+1:2d}. support={sup:5d}  {pat}")

    return True


if __name__ == "__main__":
    passed = 0
    failed = 0
    try:
        test_sample_database()
        passed += 1
    except Exception as e:
        print(f"\nTEST 1 FAILED: {e}")
        failed += 1

    try:
        test_full_train()
        passed += 1
    except Exception as e:
        print(f"\nTEST 2 FAILED: {e}")
        failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
    print(f"{'='*60}")
