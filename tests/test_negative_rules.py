"""
Tests for src/negative_rules.py.

1. Planted X and not M implies Y is recovered; the M-present reverse is not.
2. For every (X, M), len(neg)+len(pos) equals examples containing X.
3. Random-noise data yields no rules after BH correction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "mining")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from negative_rules import add_seq_id, mine_negative_rules, xm_partition


def _examples_from_sequences(sequences):
    rows = []
    for sid, seq in enumerate(sequences):
        seq = tuple(seq)
        for k in range(2, len(seq)):
            rows.append({
                "prefix": seq[:k],
                "next": seq[k],
                "prefix_len": k,
                "full_seq": seq,
                "seq_id": f"s{sid}",
                "dup_count": 1,
            })
    return pd.DataFrame(rows)


def test_planted_rule_found_reverse_not():
    """X={A}, M=B, Y=C is planted; next=D only when B is already in the prefix.
    Prefix length is >= 2 by prepending 'Z'.
    """
    planted = [["Z", "A", "C", "E", "B"]] * 40
    reverse = [["Z", "B", "A", "D", "E"]] * 30
    sequences = planted + reverse
    examples = _examples_from_sequences(sequences)

    rules = mine_negative_rules(
        examples,
        sequences,
        min_x=20,
        min_seq_x=20,
        min_group=20,
        min_expect=0.6,
        min_count=5,
        min_lift=1.5,
        alpha=0.05,
    )
    assert len(rules) > 0, "planted miner returned no rules"

    planted_hits = rules[
        (rules["X"] == "A") & (rules["M"] == "B") & (rules["Y"] == "C")
    ]
    assert len(planted_hits) == 1, f"expected planted A and not B => C; got\n{rules}"

    reverse_hits = rules[
        (rules["X"] == "A") & (rules["M"] == "B") & (rules["Y"] == "D")
    ]
    assert len(reverse_hits) == 0, "must not return the M-present case Y=D"


def test_neg_plus_pos_equals_x_examples():
    sequences = [["A", "C", "E", "B"]] * 25 + [["B", "A", "D", "F"]] * 25
    examples = _examples_from_sequences(sequences)
    examples = add_seq_id(examples)

    x_list = [("A",), ("B",), ("A", "B")]
    m_list = ["A", "B", "C", "D", "E", "F"]
    for x in x_list:
        x_set = set(x)
        n_x = int(examples["prefix"].map(lambda p: x_set.issubset(p)).sum())
        if n_x == 0:
            continue
        for m in m_list:
            if m in x_set:
                continue
            neg, pos = xm_partition(examples, x, m)
            assert len(neg) + len(pos) == n_x, (
                f"X={x} M={m}: {len(neg)}+{len(pos)} != {n_x}"
            )


def test_random_noise_no_rules_after_correction():
    rng = np.random.RandomState(42)
    alphabet = [f"T{i:03d}" for i in range(20)]
    sequences = []
    for _ in range(80):
        seq = rng.choice(alphabet, size=6, replace=False).tolist()
        sequences.append(seq)
    examples = _examples_from_sequences(sequences)

    rules = mine_negative_rules(
        examples,
        sequences,
        min_x=10,
        min_seq_x=10,
        min_group=10,
        min_expect=0.6,
        min_count=5,
        min_lift=1.5,
        alpha=0.05,
    )
    assert len(rules) == 0, f"noise data produced {len(rules)} rules after BH:\n{rules}"


if __name__ == "__main__":
    test_planted_rule_found_reverse_not()
    print("PASS test_planted_rule_found_reverse_not")
    test_neg_plus_pos_equals_x_examples()
    print("PASS test_neg_plus_pos_equals_x_examples")
    test_random_noise_no_rules_after_correction()
    print("PASS test_random_noise_no_rules_after_correction")
    print("ALL THREE TESTS PASSED")
