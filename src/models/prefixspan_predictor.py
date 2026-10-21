"""
Task 3 – PrefixSpan Pattern-Based Next-Step Predictor & Validation Evaluation.

Algorithm & Architecture:
  1. Mine frequent sequential patterns from train sequences using our custom
     PrefixSpan (src/prefixspan.py) with projected databases.
  2. Extract sequential rules:
       Antecedent A -> Consequent x
       Rule confidence = support(A + [x]) / support(A)
  3. Predictor logic:
       - Given a test/val prefix, find matching candidate rules where:
         * Antecedent ends at the last item of the prefix: A[-1] == prefix[-1]
         * A[:-1] is an ordered subsequence of prefix[:-1]
         * Rule confidence >= min_conf
       - If multiple rules match:
         * Longest antecedent first (highest specificity)
         * Rank candidate consequents by rule confidence desc, support desc, alphabetical
       - Back off to Markov-2 (m=5) for any remaining/uncovered techniques or
         when no pattern rule meets min_conf.
  4. Hyperparameters tuned strictly on VALIDATION MRR.
  5. Test set is NEVER touched.

Design choices (viva):
  - Antecedent ending at prefix[-1] ensures the pattern anchors to the immediate
    attack state while leveraging multi-hop historical context.
  - Backing off to Markov-2 provides guaranteed coverage across all 173 vocab techniques.
  - No external PrefixSpan library used.
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "mining"), str(SRC / "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import config
from prefixspan import prefixspan_mine
from baselines import (
    MarkovOrder1,
    MarkovOrder2,
    build_singleton_prior,
    evaluate_baseline,
)


def is_subsequence(sub: tuple, seq: tuple) -> bool:
    """Return True if sub is an ordered subsequence of seq."""
    it = iter(seq)
    return all(x in it for x in sub)


class PrefixSpanPredictor:
    """
    Pattern-based next-step predictor backed by PrefixSpan rules with
    graceful back-off to Markov-2.
    """

    def __init__(
        self,
        min_support_pct: float = 0.01,
        min_conf: float = 0.7,
        max_len: int = 4,
        markov2_fallback: MarkovOrder2 = None,
        longest_first: bool = True,
    ):
        self.min_support_pct = min_support_pct
        self.min_conf = min_conf
        self.max_len = max_len
        self.m2 = markov2_fallback
        self.longest_first = longest_first
        self.rules = defaultdict(list)
        self.patterns = []

    def fit(self, sequences: list, weights: list):
        """Mine frequent sequential patterns and construct high-confidence rules."""
        total_w = sum(weights)
        min_sup = int(total_w * self.min_support_pct)
        self.patterns = prefixspan_mine(
            sequences, min_support=min_sup, max_len=self.max_len, weights=weights
        )
        pat_dict = {tuple(p): s for s, p in self.patterns}

        self.rules.clear()
        for p, s in pat_dict.items():
            if len(p) >= 2:
                ante = tuple(p[:-1])
                cons = p[-1]
                ante_s = pat_dict.get(ante, total_w)
                conf = s / ante_s
                if conf >= self.min_conf:
                    self.rules[ante].append((cons, conf, s))

    def predict(self, prefix: tuple) -> list:
        """
        Rank next-technique predictions:
          1. Match rules where antecedent ends at prefix[-1] and is a subsequence.
          2. Score candidate consequents by confidence (and support).
          3. Back off / fill remaining rankings with Markov-2.
        """
        m2_ranked = self.m2.predict(prefix) if self.m2 else []
        last_t = prefix[-1]

        matching_antes = [
            a for a in self.rules
            if a[-1] == last_t and is_subsequence(a[:-1], prefix[:-1])
        ]

        if not matching_antes:
            return m2_ranked

        if self.longest_first:
            max_l = max(len(a) for a in matching_antes)
            eval_antes = [a for a in matching_antes if len(a) == max_l]
        else:
            eval_antes = matching_antes

        cand_scores = {}
        cand_sups = {}
        for a in eval_antes:
            for cons, conf, s in self.rules[a]:
                if conf > cand_scores.get(cons, -1):
                    cand_scores[cons] = conf
                    cand_sups[cons] = s

        sorted_cands = sorted(
            cand_scores.keys(), key=lambda c: (-cand_scores[c], -cand_sups[c], c)
        )
        seen = set(sorted_cands)
        return sorted_cands + [t for t in m2_ranked if t not in seen]


def tune_and_evaluate_prefixspan():
    """Tune PrefixSpanPredictor on validation set, compare against all baselines."""
    processed_dir = config.BASE_DIR / "data" / "processed"
    results_dir = config.PREFIXSPAN_RESULTS_DIR

    train_df = pd.read_csv(processed_dir / "train_sequences.csv")
    train_ex = pd.read_pickle(processed_dir / "train_examples.pkl")
    val_ex = pd.read_pickle(processed_dir / "val_examples.pkl")

    seqs = [tuple(ast.literal_eval(r["sequence"])) for _, r in train_df.iterrows()]
    weights = train_df["dup_count"].tolist()

    # Pre-fit Markov fallback models (using Stage B best hyperparameters)
    singleton_counts = build_singleton_prior(config.SIGHTINGS_DATA_PATH)
    m1 = MarkovOrder1(k=0.001, fallback_counts=singleton_counts)
    m1.fit(train_ex)
    m2 = MarkovOrder2(m=5, k1=0.001, fallback_counts=singleton_counts)
    m2.fit(train_ex, markov1=m1)

    print("\n" + "=" * 70)
    print("TUNING PREFIXSPAN PREDICTOR ON VALIDATION SET")
    print("=" * 70)

    best_mrr = -1.0
    best_config = None
    tuning_rows = []

    # Compact grid: 2 supports x 3 confidences
    for sup_pct in [0.01, 0.03]:
        for conf in [0.5, 0.6, 0.7]:
            model = PrefixSpanPredictor(
                min_support_pct=sup_pct,
                min_conf=conf,
                max_len=4,
                markov2_fallback=m2,
                longest_first=True,
            )
            model.fit(seqs, weights)
            res = evaluate_baseline(model, val_ex, f"PS_sup{sup_pct}_conf{conf}")
            mrr = res["uw_mrr"]
            tuning_rows.append({
                "min_support": sup_pct,
                "min_conf": conf,
                "uw_top_1": res["uw_top_1"],
                "uw_top_3": res["uw_top_3"],
                "uw_top_5": res["uw_top_5"],
                "uw_mrr": res["uw_mrr"],
                "wt_mrr": res["wt_mrr"],
            })
            print(
                f"  sup={sup_pct:.2f}, min_conf={conf:.1f} -> "
                f"uw_top1={res['uw_top_1']:.4f}, uw_top3={res['uw_top_3']:.4f}, "
                f"uw_top5={res['uw_top_5']:.4f}, uw_MRR={mrr:.4f} | wt_MRR={res['wt_mrr']:.4f}"
            )
            if mrr > best_mrr:
                best_mrr = mrr
                best_config = (sup_pct, conf)

    print(f"\n-> Best Configuration: sup={best_config[0]}, min_conf={best_config[1]} (uw_MRR={best_mrr:.4f})")

    # Fit final tuned model
    best_model = PrefixSpanPredictor(
        min_support_pct=best_config[0],
        min_conf=best_config[1],
        max_len=4,
        markov2_fallback=m2,
        longest_first=True,
    )
    best_model.fit(seqs, weights)
    best_res = evaluate_baseline(
        best_model, val_ex, f"B5_PrefixSpan_sup{best_config[0]}_conf{best_config[1]}"
    )

    # Save tuning log
    pd.DataFrame(tuning_rows).to_csv(
        results_dir / "prefixspan_tuning_results.csv", index=False
    )

    # Load baseline results and append PrefixSpan
    baselines_df = pd.read_csv(config.BASELINES_RESULTS_DIR / "baselines_results.csv")
    new_row = {
        "model": best_res["model"],
        "uw_top_1": round(best_res["uw_top_1"], 4),
        "uw_top_3": round(best_res["uw_top_3"], 4),
        "uw_top_5": round(best_res["uw_top_5"], 4),
        "uw_mrr": round(best_res["uw_mrr"], 4),
        "wt_top_1": round(best_res["wt_top_1"], 4),
        "wt_top_3": round(best_res["wt_top_3"], 4),
        "wt_top_5": round(best_res["wt_top_5"], 4),
        "wt_mrr": round(best_res["wt_mrr"], 4),
        "seq_top_1": round(best_res["seq_top_1"], 4),
        "seq_top_3": round(best_res["seq_top_3"], 4),
        "seq_top_5": round(best_res["seq_top_5"], 4),
        "seq_mrr": round(best_res["seq_mrr"], 4),
        "macro_recall": round(best_res["macro_recall"], 4),
        "macro_f1": round(best_res["macro_f1"], 4),
        "seq_macro_recall": round(best_res["seq_macro_recall"], 4),
        "seq_macro_f1": round(best_res["seq_macro_f1"], 4),
        "wt_macro_recall": round(best_res["wt_macro_recall"], 4),
        "wt_macro_f1": round(best_res["wt_macro_f1"], 4),
    }
    all_models_df = pd.concat([baselines_df, pd.DataFrame([new_row])], ignore_index=True)
    all_models_df.to_csv(config.COMPARISON_RESULTS_DIR / "all_models_validation_results.csv", index=False)

    print("\n" + "=" * 90)
    print("ALL MODELS VALIDATION COMPARISON TABLE")
    print("=" * 90)
    print(all_models_df.to_string(index=False))

    return all_models_df, best_res


if __name__ == "__main__":
    tune_and_evaluate_prefixspan()
