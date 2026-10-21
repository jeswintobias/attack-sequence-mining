"""
STOP 3 Experiments:
  3. Validation Split-Half Cross-Validation (Half A -> Half B, Half B -> Half A)
  4. Insertion Position ('front' vs 'after_top1') and 'exceeds_base' thresholding
  5. B7 Control Experiment: Positive companion rules (X and M present -> Y)

All evaluations on VALIDATION only. Test set is NEVER touched.
All results saved to results/ directory.
"""

from __future__ import annotations

import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "mining"), str(SRC / "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from baselines import evaluate_baseline, evaluate_predictions
from negative_predictor import (
    _load_train_val,
    _fit_b4_b5,
    collect_tested_rules,
    select_rules,
    parse_x,
    NegativePredictor,
)
from negative_rules import benjamini_hochberg


# ── Helper for Base Top Score ─────────────────────────────────────────────────

def get_base_top_score(base_model, prefix: tuple) -> float:
    """Return confidence/probability score of the top candidate from the base model."""
    last_t = prefix[-1]
    matching_antes = [
        a for a in base_model.rules
        if a[-1] == last_t and all(x in iter(prefix[:-1]) for x in a[:-1])
    ]
    if matching_antes:
        max_l = max(len(a) for a in matching_antes)
        eval_antes = [a for a in matching_antes if len(a) == max_l]
        scores = [conf for a in eval_antes for cons, conf, s in base_model.rules[a]]
        return max(scores) if scores else 0.0

    # Back off to Markov-2:
    m2 = base_model.m2
    if len(prefix) >= 2:
        ctx = (prefix[-2], prefix[-1])
        if ctx in m2.trans2 and sum(m2.trans2[ctx].values()) >= m2.m:
            total_obs = sum(m2.trans2[ctx].values())
            V = len(m2.vocab)
            max_c = max(m2.trans2[ctx].values())
            return (max_c + m2.k1) / (total_obs + m2.k1 * V)

    # Back off to Markov-1:
    m1 = m2.markov1 if m2 else None
    if m1 and prefix[-1] in m1.trans:
        total_obs = sum(m1.trans[prefix[-1]].values())
        V = len(m1.vocab)
        max_c = max(m1.trans[prefix[-1]].values())
        return (max_c + m1.k) / (total_obs + m1.k * V)

    # Unigram prior:
    total_u = sum(m2.fallback_counts.values()) if m2 else 0
    max_u = max(m2.fallback_counts.values()) if total_u else 0
    return max_u / total_u if total_u else 0.0


# ── Enhanced Predictor with Position & Threshold Options ──────────────────────

class FlexibleNegativePredictor:
    def __init__(
        self,
        rules: pd.DataFrame,
        base_model,
        n_inject: int = 1,
        position: str = "front",               # 'front' or 'after_top1'
        inject_if_exceeds_base: bool = False,
    ):
        self.base_model = base_model
        self.n_inject = int(n_inject)
        self.position = position
        self.inject_if_exceeds_base = inject_if_exceeds_base
        self.rule_rows = []
        for _, r in rules.iterrows():
            self.rule_rows.append({
                "X": parse_x(r["X"]),
                "M": r["M"],
                "Y": r["Y"],
                "conf_neg": float(r["conf_neg"]),
                "count": int(r["count"]),
            })

    def fired_rules(self, prefix: tuple) -> list:
        pset = set(prefix)
        return [r for r in self.rule_rows if r["X"].issubset(pset) and r["M"] not in pset]

    def predict(self, prefix: tuple) -> list:
        base = self.base_model.predict(prefix)
        fired = self.fired_rules(prefix)
        if not fired:
            return base

        base_top_score = get_base_top_score(self.base_model, prefix) if self.inject_if_exceeds_base else 0.0

        best = {}
        for r in fired:
            y = r["Y"]
            if self.inject_if_exceeds_base and r["conf_neg"] <= base_top_score:
                continue
            cur = best.get(y)
            if cur is None or r["conf_neg"] > cur[0] or (
                r["conf_neg"] == cur[0] and r["count"] > cur[1]
            ):
                best[y] = (r["conf_neg"], r["count"])

        if not best:
            return base

        ranked = sorted(best.keys(), key=lambda y: (-best[y][0], -best[y][1], y))
        inject = ranked[: self.n_inject]
        seen = set(inject)

        if self.position == "after_top1":
            if not base:
                return inject
            top1 = base[0]
            if top1 in seen:
                # If base top1 was also in inject, put it first then remaining inject
                rem_inject = [t for t in inject if t != top1]
                return [top1] + rem_inject + [t for t in base[1:] if t not in seen]
            else:
                return [top1] + inject + [t for t in base[1:] if t not in seen]
        else:
            # 'front'
            return inject + [t for t in base if t not in seen]


# ── Item 3: Split-Half Validation ─────────────────────────────────────────────

def run_split_half_experiment(tested, b5, val_ex):
    print("\n" + "=" * 70)
    print("ITEM 3: SPLIT-HALF CROSS-VALIDATION (SEED 42)")
    print("=" * 70)

    # Group val_ex by unique sequence ID
    val_seq_ids = sorted(val_ex["seq_id"].unique())
    n_seqs = len(val_seq_ids)
    rng = np.random.RandomState(42)
    shuffled_ids = rng.permutation(val_seq_ids)

    half_n = n_seqs // 2
    ids_a = set(shuffled_ids[:half_n])
    ids_b = set(shuffled_ids[half_n:])

    ex_a = val_ex[val_ex["seq_id"].isin(ids_a)].copy()
    ex_b = val_ex[val_ex["seq_id"].isin(ids_b)].copy()
    print(f"Total val sequences: {n_seqs} | Half A: {len(ids_a)} seqs ({len(ex_a)} ex) | Half B: {len(ids_b)} seqs ({len(ex_b)} ex)")

    grid_min_expect = [0.6, 0.8]
    grid_min_count = [5, 10]
    grid_min_lift = [1.5, 2.0]
    grid_alpha = [0.05, 0.01]
    grid_n_inject = [1, 2, 3]

    def _tune_on_half(ex_tune):
        best_mrr = -1.0
        best_cfg = None
        for min_expect in grid_min_expect:
            for min_count in grid_min_count:
                for min_lift in grid_min_lift:
                    for alpha in grid_alpha:
                        rules = select_rules(tested, min_expect, min_count, min_lift, alpha)
                        for n_inject in grid_n_inject:
                            model = NegativePredictor(rules, b5, n_inject=n_inject)
                            res = evaluate_baseline(model, ex_tune, "tune")
                            if res["uw_mrr"] > best_mrr:
                                best_mrr = res["uw_mrr"]
                                best_cfg = {
                                    "min_expect": min_expect,
                                    "min_count": min_count,
                                    "min_lift": min_lift,
                                    "alpha": alpha,
                                    "n_inject": n_inject,
                                    "n_rules": len(rules),
                                    "tune_mrr": best_mrr,
                                }
        return best_cfg

    print("Tuning on Half A...")
    best_a = _tune_on_half(ex_a)
    rules_a = select_rules(tested, best_a["min_expect"], best_a["min_count"], best_a["min_lift"], best_a["alpha"])
    model_a = NegativePredictor(rules_a, b5, n_inject=best_a["n_inject"])
    res_b = evaluate_baseline(model_a, ex_b, "HalfA_tuned_tested_on_HalfB")

    print("Tuning on Half B...")
    best_b = _tune_on_half(ex_b)
    rules_b = select_rules(tested, best_b["min_expect"], best_b["min_count"], best_b["min_lift"], best_b["alpha"])
    model_b = NegativePredictor(rules_b, b5, n_inject=best_b["n_inject"])
    res_a = evaluate_baseline(model_b, ex_a, "HalfB_tuned_tested_on_HalfA")

    rows = [
        {
            "tune_split": "Half A",
            "test_split": "Half B",
            "min_expect": best_a["min_expect"],
            "min_count": best_a["min_count"],
            "min_lift": best_a["min_lift"],
            "alpha": best_a["alpha"],
            "n_inject": best_a["n_inject"],
            "n_rules": best_a["n_rules"],
            "tune_uw_mrr": round(best_a["tune_mrr"], 4),
            "test_uw_top1": round(res_b["uw_top_1"], 4),
            "test_uw_top3": round(res_b["uw_top_3"], 4),
            "test_uw_top5": round(res_b["uw_top_5"], 4),
            "test_uw_mrr": round(res_b["uw_mrr"], 4),
            "test_wt_top1": round(res_b["wt_top_1"], 4),
            "test_wt_mrr": round(res_b["wt_mrr"], 4),
        },
        {
            "tune_split": "Half B",
            "test_split": "Half A",
            "min_expect": best_b["min_expect"],
            "min_count": best_b["min_count"],
            "min_lift": best_b["min_lift"],
            "alpha": best_b["alpha"],
            "n_inject": best_b["n_inject"],
            "n_rules": best_b["n_rules"],
            "tune_uw_mrr": round(best_b["tune_mrr"], 4),
            "test_uw_top1": round(res_a["uw_top_1"], 4),
            "test_uw_top3": round(res_a["uw_top_3"], 4),
            "test_uw_top5": round(res_a["uw_top_5"], 4),
            "test_uw_mrr": round(res_a["uw_mrr"], 4),
            "test_wt_top1": round(res_a["wt_top_1"], 4),
            "test_wt_mrr": round(res_a["wt_mrr"], 4),
        },
    ]
    df_split = pd.DataFrame(rows)
    df_split.to_csv(config.NEGATIVE_RULES_RESULTS_DIR / "validation_split_half_sensitivity.csv", index=False)
    print(df_split.to_string(index=False))
    return df_split


# ── Item 4: Insertion Position & Exceeds-Base Threshold ───────────────────────

def run_insertion_position_experiment(tested, b5, val_ex):
    print("\n" + "=" * 70)
    print("ITEM 4: INSERTION POSITION & THRESHOLD GRID")
    print("=" * 70)

    # Use the best miner parameters: expect=0.6, count=5, lift=1.5, alpha=0.05
    rules = select_rules(tested, min_expect=0.6, min_count=5, min_lift=1.5, alpha=0.05)

    configs = [
        {"position": "front", "inject_if_exceeds_base": False, "n_inject": 3, "label": "B6_front_unconditional (original)"},
        {"position": "front", "inject_if_exceeds_base": True,  "n_inject": 3, "label": "B6_front_exceeds_base"},
        {"position": "after_top1", "inject_if_exceeds_base": False, "n_inject": 3, "label": "B6_after_top1_unconditional"},
        {"position": "after_top1", "inject_if_exceeds_base": True,  "n_inject": 3, "label": "B6_after_top1_exceeds_base"},
    ]

    records = []
    for cfg in configs:
        model = FlexibleNegativePredictor(
            rules=rules,
            base_model=b5,
            n_inject=cfg["n_inject"],
            position=cfg["position"],
            inject_if_exceeds_base=cfg["inject_if_exceeds_base"],
        )
        res = evaluate_baseline(model, val_ex, cfg["label"])
        records.append({
            "model_variant": cfg["label"],
            "position": cfg["position"],
            "inject_if_exceeds_base": cfg["inject_if_exceeds_base"],
            "uw_top1": round(res["uw_top_1"], 4),
            "uw_top3": round(res["uw_top_3"], 4),
            "uw_top5": round(res["uw_top_5"], 4),
            "uw_mrr": round(res["uw_mrr"], 4),
            "wt_top1": round(res["wt_top_1"], 4),
            "wt_top3": round(res["wt_top_3"], 4),
            "wt_top5": round(res["wt_top_5"], 4),
            "wt_mrr": round(res["wt_mrr"], 4),
        })

    df_pos = pd.DataFrame(records)
    df_pos.to_csv(config.NEGATIVE_RULES_RESULTS_DIR / "validation_insertion_position_sensitivity.csv", index=False)
    print(df_pos.to_string(index=False))
    return df_pos


# ── Item 5: Control Experiment B7 (X and M present -> Y) ──────────────────────

def collect_positive_control_rules(examples, sequences, min_x=20, min_seq_x=20, min_group=20, min_expect=0.6, min_count=5):
    """
    Control experiment miner: rules of the form X and M PRESENT implies next = Y.
    Uses identical Fisher exact test structure and count floors.
    """
    prefixes = [frozenset(p) for p in examples["prefix"].tolist()]
    labels = examples["next"].tolist()
    seq_sets = [frozenset(s) for s in sequences]

    item_to_ex = defaultdict(list)
    for i, pset in enumerate(prefixes):
        for t in pset:
            item_to_ex[t].append(i)

    item_to_seq = defaultdict(list)
    for sid, sset in enumerate(seq_sets):
        for t in sset:
            item_to_seq[t].append(sid)

    items = sorted(item_to_ex.keys())
    x_candidates = []
    x_example_idx = {}
    x_seq_idx = {}

    for a in items:
        idx = np.fromiter(item_to_ex[a], dtype=int, count=len(item_to_ex[a]))
        if len(idx) >= min_x:
            key = (a,)
            x_candidates.append(key)
            x_example_idx[key] = idx
            x_seq_idx[key] = np.fromiter(item_to_seq.get(a, []), dtype=int, count=len(item_to_seq.get(a, [])))

    from itertools import combinations
    freq1 = [a for a, in x_candidates]
    for a, b in combinations(freq1, 2):
        arr_a = item_to_ex[a]
        arr_b = item_to_ex[b]
        inter = np.intersect1d(arr_a, arr_b, assume_unique=True)
        if len(inter) >= min_x:
            key = tuple(sorted((a, b)))
            x_candidates.append(key)
            x_example_idx[key] = inter
            x_seq_idx[key] = np.intersect1d(item_to_seq.get(a, []), item_to_seq.get(b, []), assume_unique=True)

    tech_counts = Counter()
    for s in seq_sets:
        tech_counts.update(s)
    m_universe = sorted(tech_counts.keys())

    tested = []
    from scipy.stats import fisher_exact

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
            n_pos = int(present.sum())
            n_neg = int((~present).sum())
            if n_pos < min_group or n_neg == 0:
                continue

            y_pos = y_arr[present]
            y_neg = y_arr[~present]
            pos_counts = Counter(y_pos.tolist())
            neg_counts = Counter(y_neg.tolist())

            for y, c in pos_counts.items():
                if c < min_count:
                    continue
                conf_pos = c / n_pos
                p_y_given_x = y_all[y] / n_x
                if p_y_given_x <= 0:
                    continue
                lift = conf_pos / p_y_given_x
                b = c
                a = neg_counts.get(y, 0)
                d = n_pos - b
                cc = n_neg - a
                table = [[b, a], [d, cc]]
                _, p_raw = fisher_exact(table, alternative="two-sided")
                tested.append({
                    "X": "|".join(x),
                    "M": m,
                    "Y": y,
                    "n_pos": n_pos,
                    "count": int(c),
                    "conf_pos": conf_pos,
                    "p_Y_given_X": p_y_given_x,
                    "lift": lift,
                    "p_raw": float(p_raw),
                    "expect": expect,
                })

    return pd.DataFrame(tested)


def select_positive_rules(tested, min_expect, min_count, min_lift, alpha):
    if tested is None or len(tested) == 0:
        return pd.DataFrame()
    sub = tested[(tested["expect"] >= min_expect) & (tested["count"] >= min_count)].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    sub["p_adj"] = benjamini_hochberg(sub["p_raw"].tolist())
    return sub[(sub["lift"] >= min_lift) & (sub["p_adj"] <= alpha)].copy()


class PositivePredictor:
    def __init__(self, rules: pd.DataFrame, base_model, n_inject: int = 1):
        self.base_model = base_model
        self.n_inject = int(n_inject)
        self.rule_rows = []
        for _, r in rules.iterrows():
            self.rule_rows.append({
                "X": parse_x(r["X"]),
                "M": r["M"],
                "Y": r["Y"],
                "conf_pos": float(r["conf_pos"]),
                "count": int(r["count"]),
            })

    def fired_rules(self, prefix: tuple) -> list:
        pset = set(prefix)
        # Firing condition for positive control: X AND M are BOTH present in prefix
        return [r for r in self.rule_rows if r["X"].issubset(pset) and r["M"] in pset]

    def predict(self, prefix: tuple) -> list:
        base = self.base_model.predict(prefix)
        fired = self.fired_rules(prefix)
        if not fired:
            return base
        best = {}
        for r in fired:
            y = r["Y"]
            cur = best.get(y)
            if cur is None or r["conf_pos"] > cur[0] or (
                r["conf_pos"] == cur[0] and r["count"] > cur[1]
            ):
                best[y] = (r["conf_pos"], r["count"])
        ranked = sorted(best.keys(), key=lambda y: (-best[y][0], -best[y][1], y))
        inject = ranked[: self.n_inject]
        seen = set(inject)
        return inject + [t for t in base if t not in seen]


def run_positive_control_experiment(train_ex, seqs, b5, val_ex):
    print("\n" + "=" * 70)
    print("ITEM 5: B7 CONTROL EXPERIMENT (POSITIVE COMPANION RULES: X and M -> Y)")
    print("=" * 70)

    print("Collecting tested positive companion rules on train...")
    tested_pos = collect_positive_control_rules(train_ex, seqs, min_x=20, min_seq_x=20, min_group=20, min_expect=0.6, min_count=5)
    print(f"  tested positive (X,M,Y) rows: {len(tested_pos)}")

    grid_min_expect = [0.6, 0.8]
    grid_min_count = [5, 10]
    grid_min_lift = [1.5, 2.0]
    grid_alpha = [0.05, 0.01]
    grid_n_inject = [1, 2, 3]

    best_mrr = -1.0
    best_cfg = None
    for min_expect in grid_min_expect:
        for min_count in grid_min_count:
            for min_lift in grid_min_lift:
                for alpha in grid_alpha:
                    rules = select_positive_rules(tested_pos, min_expect, min_count, min_lift, alpha)
                    for n_inject in grid_n_inject:
                        model = PositivePredictor(rules, b5, n_inject=n_inject)
                        res = evaluate_baseline(model, val_ex, "B7_tune")
                        if res["uw_mrr"] > best_mrr:
                            best_mrr = res["uw_mrr"]
                            best_cfg = {
                                "min_expect": min_expect,
                                "min_count": min_count,
                                "min_lift": min_lift,
                                "alpha": alpha,
                                "n_inject": n_inject,
                                "n_rules": len(rules),
                                "uw_mrr": best_mrr,
                                "res": res,
                            }

    print(f"B7 Best Config: expect={best_cfg['min_expect']}, count={best_cfg['min_count']}, lift={best_cfg['min_lift']}, alpha={best_cfg['alpha']}, n_inject={best_cfg['n_inject']} (rules={best_cfg['n_rules']}, uw_MRR={best_cfg['uw_mrr']:.4f})")
    
    # Save best B7 results side-by-side with B4, B5, B6
    res_b7 = best_cfg["res"]
    comparison_rows = [
        {"model": "B4_Markov2_m5", "uw_top1": 0.2839, "uw_top3": 0.5191, "uw_top5": 0.6538, "uw_mrr": 0.4415, "wt_top1": 0.4850, "wt_top3": 0.7324, "wt_top5": 0.8097, "wt_mrr": 0.6288},
        {"model": "B5_PrefixSpan_sup0.01_conf0.7", "uw_top1": 0.2824, "uw_top3": 0.5180, "uw_top5": 0.6567, "uw_mrr": 0.4400, "wt_top1": 0.5561, "wt_top3": 0.7367, "wt_top5": 0.8159, "wt_mrr": 0.6658},
        {"model": "B6_NegativeRules (X and not M -> Y)", "uw_top1": 0.2965, "uw_top3": 0.5382, "uw_top5": 0.6913, "uw_mrr": 0.4612, "wt_top1": 0.3400, "wt_top3": 0.5493, "wt_top5": 0.8273, "wt_mrr": 0.5154},
        {"model": f"B7_PositiveControl (X and M -> Y, inj={best_cfg['n_inject']})", "uw_top1": round(res_b7["uw_top_1"], 4), "uw_top3": round(res_b7["uw_top_3"], 4), "uw_top5": round(res_b7["uw_top_5"], 4), "uw_mrr": round(res_b7["uw_mrr"], 4), "wt_top1": round(res_b7["wt_top_1"], 4), "wt_top3": round(res_b7["wt_top_3"], 4), "wt_top5": round(res_b7["wt_top_5"], 4), "wt_mrr": round(res_b7["wt_mrr"], 4)},
    ]
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(config.NEGATIVE_RULES_RESULTS_DIR / "validation_b7_positive_control.csv", index=False)
    print("\n── SIDE-BY-SIDE MODEL COMPARISON TABLE (VALIDATION) ──")
    print(df_comp.to_string(index=False))
    return df_comp


# ── Main Entry ────────────────────────────────────────────────────────────────

def main():
    train_df, train_ex, val_ex, seqs, weights = _load_train_val()
    m2, b5 = _fit_b4_b5(train_ex, seqs, weights)

    print("Collecting negative rules on train...")
    tested = collect_tested_rules(train_ex, seqs, min_x=20, min_seq_x=20, min_group=20, min_expect=0.6, min_count=5)

    # 3. Split-half validation
    run_split_half_experiment(tested, b5, val_ex)

    # 4. Insertion position experiment
    run_insertion_position_experiment(tested, b5, val_ex)

    # 5. Positive control experiment B7
    run_positive_control_experiment(train_ex, seqs, b5, val_ex)


if __name__ == "__main__":
    main()
