"""
B6: PrefixSpan (B5) ranking with negative-rule injection.

predict(prefix):
  base = B5 ranking (PrefixSpan rules, Markov-2 back-off, sup=0.01, conf=0.7)
  fired = rules where X subset of prefix and M not in prefix
  if none fire, return base
  rank Y by max conf_neg (then count, then Y id)
  inject the first n_inject of that list
  return inject + [t for t in base if t not in inject]

Tune miner thresholds and n_inject on validation unweighted MRR.
Never loads a test file. Examples counted once per unique sequence.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "mining"), str(SRC / "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import config
from baselines import (
    MarkovOrder1,
    MarkovOrder2,
    build_singleton_prior,
    evaluate_baseline,
    evaluate_predictions,
)
from prefixspan_predictor import PrefixSpanPredictor
from negative_rules import (
    add_seq_id,
    collect_tested_rules,
    select_rules,
)


B5_SUP = 0.01
B5_CONF = 0.7
B4_M = 5
B4_K = 0.001


def parse_x(x_str: str) -> frozenset:
    return frozenset(x_str.split("|"))


class NegativePredictor:
    def __init__(self, rules: pd.DataFrame, base_model, n_inject: int = 1):
        self.base_model = base_model
        self.n_inject = int(n_inject)
        self.rule_rows = []
        for _, r in rules.iterrows():
            self.rule_rows.append({
                "X": parse_x(r["X"]),
                "M": r["M"],
                "Y": r["Y"],
                "conf_neg": float(r["conf_neg"]),
                "count": int(r["count"]),
            })

    def fired_rules(self, prefix) -> list:
        pset = set(prefix)
        fired = []
        for r in self.rule_rows:
            if r["X"].issubset(pset) and r["M"] not in pset:
                fired.append(r)
        return fired

    def predict(self, prefix) -> list:
        base = self.base_model.predict(prefix)
        fired = self.fired_rules(prefix)
        if not fired:
            return base
        best = {}
        for r in fired:
            y = r["Y"]
            cur = best.get(y)
            if cur is None or r["conf_neg"] > cur[0] or (
                r["conf_neg"] == cur[0] and r["count"] > cur[1]
            ):
                best[y] = (r["conf_neg"], r["count"])
        ranked = sorted(
            best.keys(),
            key=lambda y: (-best[y][0], -best[y][1], y),
        )
        inject = ranked[: self.n_inject]
        seen = set(inject)
        return inject + [t for t in base if t not in seen]


def _load_train_val():
    processed = config.BASE_DIR / "data" / "processed"
    train_df = pd.read_csv(processed / "train_sequences.csv")
    train_ex = add_seq_id(pd.read_pickle(processed / "train_examples.pkl"))
    val_ex = add_seq_id(pd.read_pickle(processed / "val_examples.pkl"))
    seqs = [tuple(ast.literal_eval(r["sequence"])) for _, r in train_df.iterrows()]
    weights = train_df["dup_count"].tolist()
    return train_df, train_ex, val_ex, seqs, weights


def _fit_b4_b5(train_ex, seqs, weights):
    singleton_counts = build_singleton_prior(config.SIGHTINGS_DATA_PATH)
    m1 = MarkovOrder1(k=B4_K, fallback_counts=singleton_counts)
    m1.fit(train_ex)
    m2 = MarkovOrder2(m=B4_M, k1=B4_K, fallback_counts=singleton_counts)
    m2.fit(train_ex, markov1=m1)
    b5 = PrefixSpanPredictor(
        min_support_pct=B5_SUP,
        min_conf=B5_CONF,
        max_len=4,
        markov2_fallback=m2,
        longest_first=True,
    )
    b5.fit(seqs, weights)
    return m2, b5


def _metrics_from_pred_lists(pred_lists, examples, model_name):
    records = []
    for preds, (_, r) in zip(pred_lists, examples.iterrows()):
        scores = evaluate_predictions(preds, r["next"])
        scores["dup_count"] = r["dup_count"]
        records.append(scores)
    df_res = pd.DataFrame(records)
    out = {"model": model_name}
    for col in ["top_1", "top_3", "top_5", "mrr"]:
        out[f"uw_{col}"] = float(df_res[col].mean())
        total_w = df_res["dup_count"].sum()
        out[f"wt_{col}"] = float(
            (df_res[col] * df_res["dup_count"]).sum() / total_w
        )
    return out


def _per_example_eval(model, examples):
    rows = []
    for _, r in examples.iterrows():
        preds = model.predict(r["prefix"])
        scores = evaluate_predictions(preds, r["next"])
        fired = False
        if hasattr(model, "fired_rules"):
            fired = len(model.fired_rules(r["prefix"])) > 0
        rows.append({
            "seq_id": r["seq_id"],
            "next": r["next"],
            "dup_count": r["dup_count"],
            "fired": fired,
            "top_1": scores["top_1"],
            "mrr": scores["mrr"],
            "preds": preds,
        })
    return pd.DataFrame(rows)


def _rule_fire_stats(rules: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    out = rules.copy()
    fire_n = []
    hit_n = []
    for _, rule in rules.iterrows():
        x = parse_x(rule["X"])
        m = rule["M"]
        y = rule["Y"]
        fires = 0
        hits = 0
        for _, e in examples.iterrows():
            pset = set(e["prefix"])
            if x.issubset(pset) and m not in pset:
                fires += 1
                if e["next"] == y:
                    hits += 1
        fire_n.append(fires)
        hit_n.append(hits)
    out["val_fire_count"] = fire_n
    out["val_hit_count"] = hit_n
    out["val_hit_rate"] = [
        (h / f if f else 0.0) for h, f in zip(hit_n, fire_n)
    ]
    return out


def _paired_bootstrap_mrr(seq_ids, mrr_b5, mrr_b6, n_resamples=1000, seed=42):
    """
    Resample unique sequences. Sequence MRR is the mean of its example MRRs.
    Returns mean(B6-B5), 2.5th and 97.5th percentiles.
    """
    df = pd.DataFrame({"seq_id": seq_ids, "mrr_b5": mrr_b5, "mrr_b6": mrr_b6})
    per_seq = df.groupby("seq_id")[["mrr_b5", "mrr_b6"]].mean()
    ids = per_seq.index.to_numpy()
    rng = np.random.RandomState(seed)
    n = len(ids)
    diffs = np.empty(n_resamples, dtype=float)
    b5 = per_seq["mrr_b5"].to_numpy()
    b6 = per_seq["mrr_b6"].to_numpy()
    id_to_i = {sid: i for i, sid in enumerate(ids)}
    idx = np.arange(n)
    for t in range(n_resamples):
        draw = rng.choice(idx, size=n, replace=True)
        diffs[t] = b6[draw].mean() - b5[draw].mean()
    return {
        "n_sequences": n,
        "n_resamples": n_resamples,
        "mean_diff_b6_minus_b5": float(diffs.mean()),
        "ci_2.5": float(np.percentile(diffs, 2.5)),
        "ci_97.5": float(np.percentile(diffs, 97.5)),
        "seed": seed,
    }


def run_negative_experiment():
    results_dir = config.NEGATIVE_RULES_RESULTS_DIR
    train_df, train_ex, val_ex, seqs, weights = _load_train_val()
    m2, b5 = _fit_b4_b5(train_ex, seqs, weights)

    print("Collecting tested negative rules on train (unique sequences)...")
    tested = collect_tested_rules(
        train_ex, seqs,
        min_x=20, min_seq_x=20, min_group=20,
        min_expect=0.6, min_count=5,
    )
    print(f"  tested (X,M,Y) rows at loosest grid floors: {len(tested)}")

    grid_min_expect = [0.6, 0.8]
    grid_min_count = [5, 10]
    grid_min_lift = [1.5, 2.0]
    grid_alpha = [0.05, 0.01]
    grid_n_inject = [1, 2, 3]

    grid_rows = []
    best_mrr = -1.0
    best = None
    config_i = 0
    for min_expect in grid_min_expect:
        for min_count in grid_min_count:
            for min_lift in grid_min_lift:
                for alpha in grid_alpha:
                    rules = select_rules(
                        tested, min_expect, min_count, min_lift, alpha
                    )
                    n_rules = len(rules)
                    for n_inject in grid_n_inject:
                        config_i += 1
                        model = NegativePredictor(rules, b5, n_inject=n_inject)
                        res = evaluate_baseline(
                            model, val_ex, f"B6_cfg{config_i}"
                        )
                        row = {
                            "config_id": config_i,
                            "min_expect": min_expect,
                            "min_count": min_count,
                            "min_lift": min_lift,
                            "alpha": alpha,
                            "n_inject": n_inject,
                            "n_rules": n_rules,
                            "uw_top_1": res["uw_top_1"],
                            "uw_top_3": res["uw_top_3"],
                            "uw_top_5": res["uw_top_5"],
                            "uw_mrr": res["uw_mrr"],
                            "wt_top_1": res["wt_top_1"],
                            "wt_top_3": res["wt_top_3"],
                            "wt_top_5": res["wt_top_5"],
                            "wt_mrr": res["wt_mrr"],
                            "seq_top_1": res["seq_top_1"],
                            "seq_top_3": res["seq_top_3"],
                            "seq_top_5": res["seq_top_5"],
                            "seq_mrr": res["seq_mrr"],
                            "macro_recall": res["macro_recall"],
                            "macro_f1": res["macro_f1"],
                            "seq_macro_recall": res["seq_macro_recall"],
                            "seq_macro_f1": res["seq_macro_f1"],
                            "wt_macro_recall": res["wt_macro_recall"],
                            "wt_macro_f1": res["wt_macro_f1"],
                        }
                        grid_rows.append(row)
                        print(
                            f"  [{config_i:02d}/48] expect={min_expect} "
                            f"count={min_count} lift={min_lift} alpha={alpha} "
                            f"n_inject={n_inject} n_rules={n_rules} "
                            f"uw_MRR={res['uw_mrr']:.4f}"
                        )
                        if res["uw_mrr"] > best_mrr:
                            best_mrr = res["uw_mrr"]
                            best = {
                                "row": row,
                                "rules": rules,
                                "n_inject": n_inject,
                            }

    df_grid = pd.DataFrame(grid_rows)
    df_grid.to_csv(results_dir / "validation_negative_rule_grid_search.csv", index=False)
    print(f"\nChosen config by uw_MRR: {best['row']}")

    chosen_rules = best["rules"]
    b6 = NegativePredictor(chosen_rules, b5, n_inject=best["n_inject"])

    res_b4 = evaluate_baseline(m2, val_ex, "B4_Markov2_m5")
    res_b5 = evaluate_baseline(b5, val_ex, "B5_PrefixSpan_sup0.01_conf0.7")
    res_b6 = evaluate_baseline(b6, val_ex, "B6_NegativeRules")

    val_rows = []
    for r in (res_b4, res_b5, res_b6):
        val_rows.append({
            "model": r["model"],
            "uw_top_1": round(r["uw_top_1"], 4),
            "uw_top_3": round(r["uw_top_3"], 4),
            "uw_top_5": round(r["uw_top_5"], 4),
            "uw_mrr": round(r["uw_mrr"], 4),
            "wt_top_1": round(r["wt_top_1"], 4),
            "wt_top_3": round(r["wt_top_3"], 4),
            "wt_top_5": round(r["wt_top_5"], 4),
            "wt_mrr": round(r["wt_mrr"], 4),
            "seq_top_1": round(r["seq_top_1"], 4),
            "seq_top_3": round(r["seq_top_3"], 4),
            "seq_top_5": round(r["seq_top_5"], 4),
            "seq_mrr": round(r["seq_mrr"], 4),
            "macro_recall": round(r["macro_recall"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "seq_macro_recall": round(r["seq_macro_recall"], 4),
            "seq_macro_f1": round(r["seq_macro_f1"], 4),
            "wt_macro_recall": round(r["wt_macro_recall"], 4),
            "wt_macro_f1": round(r["wt_macro_f1"], 4),
        })

    # Fired vs non-fired (examples)
    b5_ex = _per_example_eval(b5, val_ex)
    b6_ex = _per_example_eval(b6, val_ex)
    assert len(b5_ex) == len(b6_ex)
    fired_mask = b6_ex["fired"].to_numpy()
    fire_rate = float(fired_mask.mean())
    split_rows = []
    for label, mask in (
        ("fired", fired_mask),
        ("non_fired", ~fired_mask),
        ("all", np.ones(len(b6_ex), dtype=bool)),
    ):
        if mask.sum() == 0:
            continue
        split_rows.append({
            "subset": label,
            "n_examples": int(mask.sum()),
            "fire_rate": fire_rate if label == "all" else (
                1.0 if label == "fired" else 0.0
            ),
            "b5_uw_top_1": float(b5_ex.loc[mask, "top_1"].mean()),
            "b5_uw_mrr": float(b5_ex.loc[mask, "mrr"].mean()),
            "b6_uw_top_1": float(b6_ex.loc[mask, "top_1"].mean()),
            "b6_uw_mrr": float(b6_ex.loc[mask, "mrr"].mean()),
        })
    df_split = pd.DataFrame(split_rows)

    df_val = pd.DataFrame(val_rows)
    # append fire-split as extra section via a second csv block in same file
    # Store overall table and fired table together with a source tag
    df_val.insert(0, "section", "overall")
    df_split_out = df_split.copy()
    df_split_out.insert(0, "section", "fired_vs_nonfired")
    pd.concat([df_val, df_split_out], ignore_index=True).to_csv(
        results_dir / "validation_negative_rule_metrics.csv", index=False
    )
    df_val.to_csv(results_dir / "validation_negative_rule_summary.csv", index=False)
    df_split.to_csv(results_dir / "validation_negative_rule_fire_breakdown.csv", index=False)

    top30 = chosen_rules.head(30).copy()
    top30 = _rule_fire_stats(top30, val_ex)
    top30.to_csv(results_dir / "negative_rules_top30.csv", index=False)

    boot = _paired_bootstrap_mrr(
        b6_ex["seq_id"].tolist(),
        b5_ex["mrr"].to_numpy(),
        b6_ex["mrr"].to_numpy(),
        n_resamples=1000,
        seed=config.RANDOM_SEED,
    )
    boot["b6_beats_b5_uw_mrr"] = res_b6["uw_mrr"] > res_b5["uw_mrr"]
    boot["chosen_min_expect"] = best["row"]["min_expect"]
    boot["chosen_min_count"] = best["row"]["min_count"]
    boot["chosen_min_lift"] = best["row"]["min_lift"]
    boot["chosen_alpha"] = best["row"]["alpha"]
    boot["chosen_n_inject"] = best["row"]["n_inject"]
    boot["chosen_n_rules"] = best["row"]["n_rules"]
    boot["grid_size"] = len(df_grid)
    boot["fire_rate"] = fire_rate
    pd.DataFrame([boot]).to_csv(results_dir / "validation_negative_rule_paired_bootstrap.csv", index=False)

    print("\nOverall validation:")
    print(df_val.to_string(index=False))
    print("\nFired vs non-fired:")
    print(df_split.to_string(index=False))
    print("\nBootstrap:", boot)
    return {
        "grid": df_grid,
        "best": best["row"],
        "val": df_val,
        "split": df_split,
        "boot": boot,
    }


if __name__ == "__main__":
    run_negative_experiment()
