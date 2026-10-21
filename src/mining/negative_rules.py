"""
Negative association rules of the form: X and not M implies next = Y.

X: 1 or 2 techniques, all present in the prefix.
M: one technique NOT in the prefix, but that usually accompanies X in the
   full unique sequence.
Y: the next technique.

Absence means only "not seen in the prefix so far". Sightings have no
timestamps, so these rules are not described as evasion or log tampering.

Examples are counted once per unique sequence (dup_count is ignored).
Fit on train only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def add_seq_id(examples: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable seq_id from full_seq when missing."""
    df = examples.copy()
    if "seq_id" not in df.columns:
        df["seq_id"] = df["full_seq"].map(lambda s: str(tuple(s)))
    return df


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH adjusted p-values (monotone, clipped to [0, 1])."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj_sorted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, p[order[i]] * n / rank)
        adj_sorted[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = adj_sorted
    return np.clip(out, 0.0, 1.0)


def _as_set(prefix) -> frozenset:
    return frozenset(prefix)


def _x_key(x: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(x))


def xm_partition(examples: pd.DataFrame, x: Iterable[str], m: str):
    """
    Split examples whose prefix contains X into M-absent (neg) and M-present (pos).

    Returns (neg_df, pos_df). Invariant: len(neg)+len(pos) == n examples with X.
    """
    x_set = set(x)
    mask_x = examples["prefix"].map(lambda p: x_set.issubset(p))
    ex_x = examples.loc[mask_x]
    m_in = ex_x["prefix"].map(lambda p: m in p)
    return ex_x.loc[~m_in], ex_x.loc[m_in]


def collect_tested_rules(
    examples: pd.DataFrame,
    sequences: Sequence,
    min_x: int = 20,
    min_seq_x: int = 20,
    min_group: int = 20,
    min_expect: float = 0.6,
    min_count: int = 5,
    restrict_techniques: Optional[int] = None,
) -> pd.DataFrame:
    """All (X, M, Y) Fisher tests that pass count/expect floors, with raw p."""
    examples = add_seq_id(examples).reset_index(drop=True)
    prefixes = [_as_set(p) for p in examples["prefix"].tolist()]
    labels = examples["next"].tolist()

    seq_sets = [frozenset(s) for s in sequences]

    tech_counts = Counter()
    for s in seq_sets:
        tech_counts.update(s)
    allowed = None
    if restrict_techniques is not None:
        allowed = {t for t, _ in tech_counts.most_common(restrict_techniques)}

    item_to_ex = defaultdict(list)
    for i, pset in enumerate(prefixes):
        for t in pset:
            if allowed is None or t in allowed:
                item_to_ex[t].append(i)

    item_to_seq = defaultdict(list)
    for sid, sset in enumerate(seq_sets):
        for t in sset:
            if allowed is None or t in allowed:
                item_to_seq[t].append(sid)

    items = sorted(item_to_ex.keys())
    x_candidates: List[Tuple[str, ...]] = []
    x_example_idx = {}
    x_seq_idx = {}

    def _intersect(lists: List[List[int]]) -> np.ndarray:
        if not lists:
            return np.array([], dtype=int)
        acc = None
        for lst in sorted(lists, key=len):
            arr = np.fromiter(lst, dtype=int, count=len(lst))
            acc = arr if acc is None else np.intersect1d(acc, arr, assume_unique=True)
            if len(acc) == 0:
                return acc
        return acc

    for a in items:
        idx = np.fromiter(item_to_ex[a], dtype=int, count=len(item_to_ex[a]))
        if len(idx) >= min_x:
            key = (a,)
            x_candidates.append(key)
            x_example_idx[key] = idx
            x_seq_idx[key] = np.fromiter(
                item_to_seq.get(a, []), dtype=int, count=len(item_to_seq.get(a, []))
            )

    freq1 = [a for a, in x_candidates]
    for a, b in combinations(freq1, 2):
        idx = _intersect([item_to_ex[a], item_to_ex[b]])
        if len(idx) >= min_x:
            key = _x_key((a, b))
            x_candidates.append(key)
            x_example_idx[key] = idx
            x_seq_idx[key] = _intersect(
                [item_to_seq.get(a, []), item_to_seq.get(b, [])]
            )

    m_universe = sorted(set(tech_counts.keys()) if allowed is None else allowed)
    tested = []

    for x in x_candidates:
        seq_idx = x_seq_idx[x]
        if len(seq_idx) < min_seq_x:
            continue
        ex_idx = x_example_idx[x]
        x_set = set(x)

        y_all = Counter(labels[i] for i in ex_idx)
        n_x = len(ex_idx)
        seqs_x_sets = [seq_sets[i] for i in seq_idx]
        n_seqs_x = len(seqs_x_sets)

        # prefix membership of each example for all M via boolean arrays
        ex_has = {t: np.zeros(n_x, dtype=bool) for t in m_universe}
        for j, ei in enumerate(ex_idx):
            for t in prefixes[ei]:
                if t in ex_has:
                    ex_has[t][j] = True

        y_arr = np.array([labels[i] for i in ex_idx], dtype=object)

        m_in_seq_count = Counter()
        for sset in seqs_x_sets:
            m_in_seq_count.update(sset)

        for m in m_universe:
            if m in x_set:
                continue
            expect = m_in_seq_count[m] / n_seqs_x if n_seqs_x else 0.0
            if expect < min_expect:
                continue
            present = ex_has[m]
            n_neg = int((~present).sum())
            n_pos = int(present.sum())
            if n_neg + n_pos != n_x:
                raise AssertionError("neg+pos must equal examples containing X")
            if n_neg < min_group or n_pos == 0:
                continue

            y_neg = y_arr[~present]
            y_pos = y_arr[present]
            neg_counts = Counter(y_neg.tolist())
            pos_counts = Counter(y_pos.tolist())

            for y, c in neg_counts.items():
                if c < min_count:
                    continue
                conf_neg = c / n_neg
                p_y_given_x = y_all[y] / n_x
                if p_y_given_x <= 0:
                    continue
                lift = conf_neg / p_y_given_x
                a = c
                b = pos_counts.get(y, 0)
                cc = n_neg - a
                d = n_pos - b
                table = [[a, b], [cc, d]]
                _, p_raw = fisher_exact(table, alternative="two-sided")
                tested.append({
                    "X": "|".join(x),
                    "M": m,
                    "Y": y,
                    "n_neg": n_neg,
                    "n_pos": n_pos,
                    "count": int(c),
                    "conf_neg": conf_neg,
                    "p_Y_given_X": p_y_given_x,
                    "lift": lift,
                    "p_raw": float(p_raw),
                    "expect": expect,
                })

    if not tested:
        return pd.DataFrame(columns=[
            "X", "M", "Y", "n_neg", "count", "conf_neg", "p_Y_given_X",
            "lift", "p_raw", "p_adj", "expect",
        ])
    return pd.DataFrame(tested)


def select_rules(
    tested: pd.DataFrame,
    min_expect: float,
    min_count: int,
    min_lift: float,
    alpha: float,
) -> pd.DataFrame:
    """BH-correct the tested set for this config, then apply lift and alpha."""
    cols_out = [
        "X", "M", "Y", "n_neg", "count", "conf_neg", "p_Y_given_X",
        "lift", "p_adj", "expect",
    ]
    if tested is None or len(tested) == 0:
        return pd.DataFrame(columns=cols_out)
    sub = tested[
        (tested["expect"] >= min_expect) & (tested["count"] >= min_count)
    ].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=cols_out)
    sub["p_adj"] = benjamini_hochberg(sub["p_raw"].tolist())
    keep = (sub["lift"] >= min_lift) & (sub["p_adj"] < alpha)
    out = sub.loc[keep, cols_out].sort_values(
        ["lift", "conf_neg", "count"], ascending=[False, False, False]
    ).reset_index(drop=True)
    return out


def mine_negative_rules(
    examples: pd.DataFrame,
    sequences: Sequence,
    min_x: int = 20,
    min_seq_x: int = 20,
    min_group: int = 20,
    min_expect: float = 0.6,
    min_count: int = 5,
    min_lift: float = 1.5,
    alpha: float = 0.05,
    restrict_techniques: Optional[int] = None,
) -> pd.DataFrame:
    """
    Mine X and not M implies next=Y rules on unique-sequence examples.

    sequences: unique train sequences (lists/tuples). Each counted once.
    restrict_techniques: if set, keep only this many most frequent train techs.
    """
    tested = collect_tested_rules(
        examples,
        sequences,
        min_x=min_x,
        min_seq_x=min_seq_x,
        min_group=min_group,
        min_expect=min_expect,
        min_count=min_count,
        restrict_techniques=restrict_techniques,
    )
    return select_rules(tested, min_expect, min_count, min_lift, alpha)
