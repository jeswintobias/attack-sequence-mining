"""Final 5-fold verdict on the A+B+C+D pipeline against the simple softprob model.

Run from the repository root:
  python src/eval/pipeline_final.py

Three models, same folds, same masking, paired per-sequence comparison:

  LightGBM_softprob   94-class softprob over the 511 prefix features (incumbent)
  Pipeline_base       XGBRanker over the 28 mining features only
  Pipeline_ABC        XGBRanker over mining + graph/Node2Vec (A) + tactics (B)
                      + candidate-prefix interactions (C)

D (prefix masking) is applied centrally to all three by the harness, so it is
not a variable here.

Ranking metrics and class metrics are reported separately, because the fold-0
screen showed them moving in opposite directions: the pipeline trails on MRR
while leading on macro F1. The test split is never loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"),
          str(SRC / "mining"), str(SRC / "eval")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from cv import assert_test_sealed, load_development_examples
from cv_harness import paired_compare, run_cv
from adapters import SklearnProbaAdapter
from pipeline_ranker import PipelineRankerAdapter
from lightgbm_predictor import _base_params

OUT_DIR = config.RESULTS_DIR / "pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _softprob():
    import lightgbm as lgb
    params = _base_params({"num_leaves": 31, "min_child_samples": 10,
                           "reg_lambda": 1.0}, 101)
    params["n_jobs"] = 1
    return SklearnProbaAdapter(lambda: lgb.LGBMClassifier(**params),
                               "LightGBM_softprob")


def main():
    assert_test_sealed()
    examples = load_development_examples()
    print(f"Development set: {len(examples):,} examples / "
          f"{examples['seq_id'].nunique():,} sequences. Test split not loaded.\n")

    builders = {
        "LightGBM_softprob": _softprob,
        "Pipeline_base": lambda: PipelineRankerAdapter(
            name="Pipeline_base", use_graph=False, use_tactics=False,
            use_interactions=False),
        "Pipeline_ABC": lambda: PipelineRankerAdapter(
            name="Pipeline_ABC", use_graph=True, use_tactics=True,
            use_interactions=True),
    }

    folds, seqs, pooled = [], {}, []
    for name, factory in builders.items():
        print(f"{name}:")
        fold_table, seq_scores, summary = run_cv(factory, name, examples)
        folds.append(fold_table)
        seqs[name] = seq_scores
        pooled.append(summary)

    metrics = ["seq_top_1", "seq_top_3", "seq_top_5", "seq_mrr",
               "seq_macro_recall", "seq_macro_f1"]
    table = pd.DataFrame(pooled)[["model"] + metrics + ["seq_mrr_sd"]].round(4)
    print("\n=== 5-fold means (sequence-balanced) ===")
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "pipeline_final_summary.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(
        OUT_DIR / "pipeline_final_folds.csv", index=False)

    print("\n=== Paired per-sequence MRR, bootstrap over 2,178 sequences ===")
    pairs = []
    for a, b in (("Pipeline_ABC", "Pipeline_base"),
                 ("Pipeline_ABC", "LightGBM_softprob"),
                 ("Pipeline_base", "LightGBM_softprob")):
        pairs.append(paired_compare(seqs[a], seqs[b], a, b))
    pair_table = pd.DataFrame(pairs)
    print(pair_table[["model_a", "model_b", "mean_diff", "ci_2.5", "ci_97.5",
                      "significant"]].to_string(index=False))
    pair_table.to_csv(OUT_DIR / "pipeline_final_paired.csv", index=False)

    # Where the two approaches disagree, per sequence: who wins and by how much.
    shared = seqs["Pipeline_ABC"].index.intersection(seqs["LightGBM_softprob"].index)
    diff = (seqs["Pipeline_ABC"].loc[shared, "rr"]
            - seqs["LightGBM_softprob"].loc[shared, "rr"])
    print(f"\nPer-sequence MRR difference (pipeline - softprob) over {len(diff)} sequences:")
    print(f"  pipeline strictly better on {float((diff > 0).mean()):.1%}, "
          f"worse on {float((diff < 0).mean()):.1%}, "
          f"tied on {float((diff == 0).mean()):.1%}")
    print(f"  mean gain when it wins  = {float(diff[diff > 0].mean()):+.4f}")
    print(f"  mean loss when it loses = {float(diff[diff < 0].mean()):+.4f}")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
