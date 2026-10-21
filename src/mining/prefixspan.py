"""
Custom PrefixSpan implementation – Projected-database sequential pattern mining.

This is our own implementation (no external PrefixSpan library in production).
The pip 'prefixspan' package may be imported in tests only for validation.

Algorithm:
  PrefixSpan mines frequent ordered subsequences from a sequence database using
  projected databases. For each prefix pattern, a projected database contains
  all suffixes of sequences where the prefix occurs as an ordered subsequence.

Key definitions:
  - A pattern [a, b, c] is a SUBSEQUENCE of [a, x, b, y, c, z] (order preserved,
    gaps allowed).
  - support(pattern) = number of sequences containing the pattern as a subsequence.
  - We use "sequence-level" support (each sequence counted at most once),
    weighted by dup_count.

Design choices (viva):
  - Projected databases: memory-efficient via (sid, start_pos) references.
  - Max pattern length from config (default 5).
  - Minimum support as fraction of total (dup-weighted) sequences.
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def prefixspan_mine(
    sequences: List[Tuple],
    min_support: int,
    max_len: int = 5,
    weights: Optional[List[int]] = None,
) -> List[Tuple[int, List]]:
    """
    Mine frequent sequential patterns using projected databases.

    Parameters:
        sequences: list of tuples, each a sequence of items
        min_support: absolute minimum support count (weighted)
        max_len: maximum pattern length
        weights: per-sequence weight (default: all 1s)

    Returns:
        List of (support, pattern_list) sorted by support descending.
    """
    if weights is None:
        weights = [1] * len(sequences)

    # Build initial projected database: list of (seq_index, start_position)
    # where start_position is 0 (entire sequence is the suffix)
    initial_projdb = [(i, 0) for i in range(len(sequences))]

    results = []

    def _mine(prefix: list, projdb: list):
        """Recursively mine patterns from a projected database."""
        if len(prefix) >= max_len:
            return

        # Count support for each possible next item
        item_projdbs = defaultdict(list)
        for sid, start in projdb:
            seq = sequences[sid]
            # Find all distinct items after 'start' position
            seen_items = set()
            for pos in range(start, len(seq)):
                item = seq[pos]
                if item not in seen_items:
                    seen_items.add(item)
                    item_projdbs[item].append((sid, pos + 1))

        # For each item meeting min_support, extend the prefix
        for item in sorted(item_projdbs.keys()):
            new_projdb = item_projdbs[item]

            # Compute weighted support: sum weight for each unique sid
            sid_set = set()
            sup = 0
            for sid, _ in new_projdb:
                if sid not in sid_set:
                    sid_set.add(sid)
                    sup += weights[sid]

            if sup >= min_support:
                new_prefix = prefix + [item]
                results.append((sup, list(new_prefix)))
                # Recurse with the projected database
                _mine(new_prefix, new_projdb)

    _mine([], initial_projdb)

    # Sort by support descending, then pattern alphabetically
    results.sort(key=lambda x: (-x[0], x[1]))
    return results
