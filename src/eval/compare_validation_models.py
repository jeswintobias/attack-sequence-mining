"""Build one sequence-balanced validation comparison from saved model runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

import config


def _find_path(filename: str, preferred_dir: Path) -> Path:
    p = preferred_dir / filename
    if p.exists():
        return p
    fallback = config.RESULTS_DIR / filename
    if fallback.exists():
        return fallback
    return p


def _row_from_baseline(row):
    return {
        "model": row["model"],
        "seq_top_1": row["seq_top_1"],
        "seq_top_3": row["seq_top_3"],
        "seq_top_5": row["seq_top_5"],
        "seq_mrr": row["seq_mrr"],
        "seq_macro_recall": row["seq_macro_recall"],
        "seq_macro_f1": row["seq_macro_f1"],
        "macro_label_scope": "validation-observed target labels",
    }


def build_comparison():
    rows = []

    base_path = _find_path("all_models_validation_results.csv", config.COMPARISON_RESULTS_DIR)
    base = pd.read_csv(base_path)
    rows.extend(_row_from_baseline(r) for _, r in base.iterrows())

    negative_path = _find_path("validation_negative_rule_summary.csv", config.NEGATIVE_RULES_RESULTS_DIR)
    negative = pd.read_csv(negative_path)
    b6 = negative.loc[negative["model"] == "B6_NegativeRules"].iloc[0]
    rows.append(_row_from_baseline(b6))

    grid_path = _find_path("lightgbm_validation_results.csv", config.LIGHTGBM_RESULTS_DIR)
    grid = pd.read_csv(grid_path)
    best_grid = grid.sort_values("unique_sequence_uw_mrr", ascending=False).iloc[0]
    rows.append({
        "model": f"LightGBM_Grid_config{int(best_grid['config_id'])}",
        "seq_top_1": best_grid["unique_sequence_uw_top_1"],
        "seq_top_3": best_grid["unique_sequence_uw_top_3"],
        "seq_top_5": best_grid["unique_sequence_uw_top_5"],
        "seq_mrr": best_grid["unique_sequence_uw_mrr"],
        "seq_macro_recall": best_grid["unique_sequence_macro_recall"],
        "seq_macro_f1": best_grid["unique_sequence_macro_f1"],
        "macro_label_scope": "validation-observed target labels",
    })

    halving_path = _find_path("lightgbm_halving_summary.json", config.LIGHTGBM_RESULTS_DIR)
    if halving_path.exists():
        halving = json.loads(halving_path.read_text())["external_validation_metrics"]
        rows.append({
            "model": "LightGBM_HalvingRandomSearchCV",
            "seq_top_1": halving["unique_sequence_uw_top_1"],
            "seq_top_3": halving["unique_sequence_uw_top_3"],
            "seq_top_5": halving["unique_sequence_uw_top_5"],
            "seq_mrr": halving["unique_sequence_uw_mrr"],
            "seq_macro_recall": halving["unique_sequence_macro_recall"],
            "seq_macro_f1": halving["unique_sequence_macro_f1"],
            "macro_label_scope": "validation-observed target labels",
        })

    xgb_path = _find_path("xgboost_optuna_summary.json", config.XGBOOST_RESULTS_DIR)
    if xgb_path.exists():
        xgb_metrics = json.loads(xgb_path.read_text())["external_validation_metrics"]
        rows.append({
            "model": "XGBoost_Optuna",
            "seq_top_1": xgb_metrics["unique_sequence_uw_top_1"],
            "seq_top_3": xgb_metrics["unique_sequence_uw_top_3"],
            "seq_top_5": xgb_metrics["unique_sequence_uw_top_5"],
            "seq_mrr": xgb_metrics["unique_sequence_uw_mrr"],
            "seq_macro_recall": xgb_metrics["unique_sequence_macro_recall"],
            "seq_macro_f1": xgb_metrics["unique_sequence_macro_f1"],
            "macro_label_scope": "validation-observed target labels",
        })

    frame = pd.DataFrame(rows)
    metric_cols = ["seq_top_1", "seq_top_3", "seq_top_5", "seq_mrr",
                   "seq_macro_recall", "seq_macro_f1"]
    frame[metric_cols] = frame[metric_cols].astype(float).round(6)
    out_path = config.COMPARISON_RESULTS_DIR / "all_models_validation_sequence_metrics.csv"
    frame.to_csv(out_path, index=False)
    print(frame.to_string(index=False))
    print(f"Saved: {out_path}")
    return frame


if __name__ == "__main__":
    build_comparison()
