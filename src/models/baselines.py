"""
Stage B – Next-Technique Prediction Baselines (Validation Only).

All counts come from TRAIN sequences, weighted by duplicate count.
Evaluation is on VALIDATION only. Test set is never touched.

Baselines:
  1. Majority class: predict the single most frequent next-technique label
     in the training set.
  2. Global unigram prior: from singleton rows (length == 1) in the raw CSV.
     Used only as a last fallback for completely unseen contexts.
  3. First-order Markov: P(next | last technique), with add-k smoothing.
     k is tuned on validation MRR.
  4. Second-order Markov: P(next | last two techniques), with back-off to
     first-order and then to unigram prior when context count < m.
     m is tuned on validation MRR.

Metrics: top-1, top-3, top-5 accuracy and MRR, both unweighted over unique
(prefix, next) examples and weighted by duplicate count.

Design note (viva): Ties are broken deterministically by sorting technique
IDs alphabetically, so results are reproducible.
"""

import ast
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import config
from evaluation_metrics import macro_recall_f1


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_singleton_prior(data_path: Path) -> Counter:
    """
    Build global unigram prior P(T) from singleton rows (length == 1)
    in the raw CSV, with subtechniques collapsed to parent.
    """
    df = pd.read_csv(data_path)
    counts = Counter()
    for raw in df["tid"].apply(ast.literal_eval):
        if len(raw) == 1:
            tech = raw[0].split(".")[0]  # collapse subtechnique
            counts[tech] += 1
    total = sum(counts.values())
    print(f"Singleton prior: {len(counts)} techniques from {total:,} singleton rows")
    return counts


def ranked_predictions(score_dict: dict) -> list:
    """
    Return technique IDs sorted by score descending, breaking ties
    by alphabetical technique ID (deterministic).
    """
    return sorted(score_dict.keys(), key=lambda t: (-score_dict[t], t))


def mask_seen(predictions: list, prefix) -> list:
    """
    Drop candidates already present in the prefix.

    First-occurrence deduplication means a technique occurs at most once per
    sequence, so the target can never be a prefix member. Controlled by
    config.MASK_SEEN_TECHNIQUES so the behaviour can be switched off and the
    unmasked numbers reproduced.
    """
    if not config.MASK_SEEN_TECHNIQUES:
        return predictions
    seen = set(prefix)
    return [t for t in predictions if t not in seen]


def evaluate_predictions(
    predictions: list,
    actual: str,
    top_k_values=(1, 3, 5),
):
    """
    Given a ranked list of predictions and the actual next technique,
    compute hit@k and reciprocal rank.
    """
    results = {}
    try:
        rank = predictions.index(actual) + 1
    except ValueError:
        rank = None

    for k in top_k_values:
        results[f"top_{k}"] = 1 if (rank is not None and rank <= k) else 0

    results["mrr"] = 1.0 / rank if rank is not None else 0.0
    return results


# ── Baseline Models ───────────────────────────────────────────────────────────

class MajorityBaseline:
    """Baseline 1: Always predicts the most frequent next-technique in train."""

    def __init__(self):
        self.majority_tech = None
        self.label_counts = None

    def fit(self, train_examples: pd.DataFrame):
        # Weight by dup_count
        weighted = train_examples.groupby("next")["dup_count"].sum()
        self.label_counts = weighted.sort_values(ascending=False)
        self.majority_tech = self.label_counts.index[0]
        total = self.label_counts.sum()
        pct = self.label_counts.iloc[0] / total * 100
        print(f"Majority baseline: {self.majority_tech} ({pct:.2f}% of train labels)")

    def predict(self, prefix):
        # Always the same prediction regardless of prefix
        return ranked_predictions({t: c for t, c in self.label_counts.items()})


class UnigramPriorBaseline:
    """Baseline 2: Global unigram prior from singleton rows."""

    def __init__(self, singleton_counts: Counter):
        self.counts = singleton_counts

    def predict(self, prefix):
        return ranked_predictions(dict(self.counts))


class MarkovOrder1:
    """
    Baseline 3: First-order Markov P(next | last technique).
    With add-k smoothing. k is tunable.
    """

    def __init__(self, k: float = 1.0, fallback_counts: Counter = None):
        self.k = k
        self.trans = defaultdict(Counter)  # trans[last][next] = weighted count
        self.vocab = set()
        self.fallback_counts = fallback_counts or Counter()

    def fit(self, train_examples: pd.DataFrame):
        self.vocab = set()
        self.trans = defaultdict(Counter)
        for _, r in train_examples.iterrows():
            last = r["prefix"][-1]
            nxt = r["next"]
            w = r["dup_count"]
            self.trans[last][nxt] += w
            self.vocab.add(last)
            self.vocab.add(nxt)
        # Add all techniques from fallback
        self.vocab.update(self.fallback_counts.keys())
        print(f"Markov-1: {len(self.trans)} contexts, vocab = {len(self.vocab)}")

    def predict(self, prefix):
        last = prefix[-1]
        V = len(self.vocab)
        if last in self.trans:
            counts = self.trans[last]
            total = sum(counts.values())
            scores = {}
            for t in self.vocab:
                scores[t] = (counts.get(t, 0) + self.k) / (total + self.k * V)
            return ranked_predictions(scores)
        else:
            # Unseen context: fall back to unigram prior
            return ranked_predictions(dict(self.fallback_counts))


class MarkovOrder2:
    """
    Baseline 4: Second-order Markov P(next | last two techniques).
    Backs off to first-order, then unigram prior, when context count < m.
    """

    def __init__(self, m: int = 5, k1: float = 1.0,
                 fallback_counts: Counter = None):
        self.m = m
        self.k1 = k1
        self.trans2 = defaultdict(Counter)  # trans2[(t_{k-1}, t_k)][next]
        self.markov1 = None
        self.fallback_counts = fallback_counts or Counter()
        self.vocab = set()

    def fit(self, train_examples: pd.DataFrame, markov1: MarkovOrder1 = None):
        self.markov1 = markov1
        self.trans2 = defaultdict(Counter)
        for _, r in train_examples.iterrows():
            pfx = r["prefix"]
            if len(pfx) >= 2:
                ctx = (pfx[-2], pfx[-1])
                nxt = r["next"]
                w = r["dup_count"]
                self.trans2[ctx][nxt] += w
        self.vocab = markov1.vocab if markov1 else set()
        print(f"Markov-2: {len(self.trans2)} bigram contexts, back-off m = {self.m}")

    def predict(self, prefix):
        if len(prefix) >= 2:
            ctx = (prefix[-2], prefix[-1])
            if ctx in self.trans2:
                total_obs = sum(self.trans2[ctx].values())
                if total_obs >= self.m:
                    counts = self.trans2[ctx]
                    V = len(self.vocab)
                    scores = {}
                    for t in self.vocab:
                        scores[t] = (counts.get(t, 0) + self.k1) / (total_obs + self.k1 * V)
                    return ranked_predictions(scores)

        # Back off to first-order
        if self.markov1 is not None:
            return self.markov1.predict(prefix)
        # Last resort: unigram prior
        return ranked_predictions(dict(self.fallback_counts))


# ── Evaluation harness ────────────────────────────────────────────────────────

def evaluate_baseline(
    model,
    examples: pd.DataFrame,
    model_name: str,
    top_k_values=(1, 3, 5),
):
    """
    Evaluate a baseline on a set of examples. Returns per-row results and
    aggregated metrics (unweighted and dup-weighted).
    """
    records = []
    for _, r in examples.iterrows():
        preds = mask_seen(model.predict(r["prefix"]), r["prefix"])
        scores = evaluate_predictions(preds, r["next"], top_k_values)
        scores["prefix_len"] = r["prefix_len"]
        scores["dup_count"] = r["dup_count"]
        scores["actual"] = r["next"]
        scores["predicted"] = preds[0] if preds else None
        scores["full_seq"] = tuple(r["full_seq"])
        records.append(scores)

    df_res = pd.DataFrame(records)

    # Aggregate: unweighted
    uw = {}
    for col in ["top_1", "top_3", "top_5", "mrr"]:
        uw[f"uw_{col}"] = df_res[col].mean()

    # Every full sequence contributes total weight 1, regardless of length.
    per_sequence_n = df_res.groupby("full_seq")["full_seq"].transform("size")
    sequence_weight = 1.0 / per_sequence_n.to_numpy(dtype=float)
    for col in ["top_1", "top_3", "top_5", "mrr"]:
        uw[f"seq_{col}"] = float(np.average(df_res[col], weights=sequence_weight))
    uw.update(macro_recall_f1(df_res["actual"], df_res["predicted"]))
    uw.update({
        f"seq_{key}": value for key, value in macro_recall_f1(
            df_res["actual"], df_res["predicted"], sequence_weight
        ).items()
    })

    # Aggregate: dup-weighted
    wt = {}
    total_w = df_res["dup_count"].sum()
    for col in ["top_1", "top_3", "top_5", "mrr"]:
        wt[f"wt_{col}"] = (df_res[col] * df_res["dup_count"]).sum() / total_w
    wt.update({
        f"wt_{key}": value for key, value in macro_recall_f1(
            df_res["actual"], df_res["predicted"], df_res["dup_count"].to_numpy()
        ).items()
    })

    # By prefix length
    by_plen = {}
    df_res["plen_group"] = df_res["prefix_len"].apply(
        lambda x: str(x) if x <= 4 else "5+"
    )
    for grp, gdf in df_res.groupby("plen_group"):
        by_plen[grp] = {}
        for col in ["top_1", "top_3", "top_5", "mrr"]:
            by_plen[grp][f"uw_{col}"] = gdf[col].mean()
            w_total = gdf["dup_count"].sum()
            by_plen[grp][f"wt_{col}"] = (
                (gdf[col] * gdf["dup_count"]).sum() / w_total if w_total > 0 else 0
            )

    return {
        "model": model_name,
        **uw,
        **wt,
        "by_prefix_len": by_plen,
        "df_results": df_res,
    }


def tune_markov1_k(
    train_ex: pd.DataFrame,
    val_ex: pd.DataFrame,
    singleton_counts: Counter,
    k_candidates=(0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0),
):
    """Tune add-k smoothing on validation MRR."""
    best_k, best_mrr = None, -1
    print("\nTuning Markov-1 add-k smoothing on validation MRR:")
    for k in k_candidates:
        m1 = MarkovOrder1(k=k, fallback_counts=singleton_counts)
        m1.fit(train_ex)
        res = evaluate_baseline(m1, val_ex, f"Markov1_k{k}")
        mrr = res["uw_mrr"]
        print(f"  k={k:6.3f}  uw_MRR={mrr:.4f}  wt_MRR={res['wt_mrr']:.4f}")
        if mrr > best_mrr:
            best_mrr = mrr
            best_k = k
    print(f"  → Best k = {best_k} (uw_MRR = {best_mrr:.4f})")
    return best_k


def tune_markov2_m(
    train_ex: pd.DataFrame,
    val_ex: pd.DataFrame,
    singleton_counts: Counter,
    best_k1: float,
    m_candidates=(1, 2, 3, 5, 10, 20, 50),
):
    """Tune back-off threshold m on validation MRR."""
    best_m, best_mrr = None, -1
    print("\nTuning Markov-2 back-off threshold m on validation MRR:")
    for m in m_candidates:
        m1 = MarkovOrder1(k=best_k1, fallback_counts=singleton_counts)
        m1.fit(train_ex)
        m2 = MarkovOrder2(m=m, k1=best_k1, fallback_counts=singleton_counts)
        m2.fit(train_ex, markov1=m1)
        res = evaluate_baseline(m2, val_ex, f"Markov2_m{m}")
        mrr = res["uw_mrr"]
        print(f"  m={m:3d}  uw_MRR={mrr:.4f}  wt_MRR={res['wt_mrr']:.4f}")
        if mrr > best_mrr:
            best_mrr = mrr
            best_m = m
    print(f"  → Best m = {best_m} (uw_MRR = {best_mrr:.4f})")
    return best_m


# ── Main entry ────────────────────────────────────────────────────────────────

def run_baselines(
    data_path: Path = None,
    results_dir: Path = None,
):
    """Run all four baselines, tune hyperparameters, report on validation."""
    if data_path is None:
        data_path = config.SIGHTINGS_DATA_PATH
    if results_dir is None:
        results_dir = config.BASELINES_RESULTS_DIR

    processed_dir = config.BASE_DIR / "data" / "processed"

    # Load examples
    train_ex = pd.read_pickle(processed_dir / "train_examples.pkl")
    val_ex = pd.read_pickle(processed_dir / "val_examples.pkl")
    print(f"Train examples: {len(train_ex):,}  |  Val examples: {len(val_ex):,}")

    # Build singleton prior (collapsed subtechniques)
    singleton_counts = build_singleton_prior(data_path)

    # ── Baseline 1: Majority ─────────────────────────────────────────────────
    majority = MajorityBaseline()
    majority.fit(train_ex)
    res_majority = evaluate_baseline(majority, val_ex, "B1_Majority")

    # ── Baseline 2: Unigram prior ─────────────────────────────────────────────
    unigram = UnigramPriorBaseline(singleton_counts)
    res_unigram = evaluate_baseline(unigram, val_ex, "B2_UnigramPrior")

    # ── Baseline 3: Markov-1 (tune k) ────────────────────────────────────────
    best_k = tune_markov1_k(train_ex, val_ex, singleton_counts)
    m1 = MarkovOrder1(k=best_k, fallback_counts=singleton_counts)
    m1.fit(train_ex)
    res_m1 = evaluate_baseline(m1, val_ex, f"B3_Markov1_k{best_k}")

    # ── Baseline 4: Markov-2 (tune m) ────────────────────────────────────────
    best_m = tune_markov2_m(train_ex, val_ex, singleton_counts, best_k)
    m1_final = MarkovOrder1(k=best_k, fallback_counts=singleton_counts)
    m1_final.fit(train_ex)
    m2 = MarkovOrder2(m=best_m, k1=best_k, fallback_counts=singleton_counts)
    m2.fit(train_ex, markov1=m1_final)
    res_m2 = evaluate_baseline(m2, val_ex, f"B4_Markov2_m{best_m}")

    # ── Compile results table ─────────────────────────────────────────────────
    all_results = [res_majority, res_unigram, res_m1, res_m2]

    # Overall table
    rows = []
    for r in all_results:
        row = {"model": r["model"]}
        for key in ["uw_top_1", "uw_top_3", "uw_top_5", "uw_mrr",
                     "wt_top_1", "wt_top_3", "wt_top_5", "wt_mrr",
                     "seq_top_1", "seq_top_3", "seq_top_5", "seq_mrr",
                     "macro_recall", "macro_f1", "seq_macro_recall",
                     "seq_macro_f1", "wt_macro_recall", "wt_macro_f1"]:
            row[key] = round(r[key], 4)
        rows.append(row)
    df_overall = pd.DataFrame(rows)

    # By prefix length table
    plen_rows = []
    for r in all_results:
        for plen, metrics in r["by_prefix_len"].items():
            prow = {"model": r["model"], "prefix_len": plen}
            for key, val in metrics.items():
                prow[key] = round(val, 4)
            plen_rows.append(prow)
    df_by_plen = pd.DataFrame(plen_rows)

    # Save
    df_overall.to_csv(results_dir / "baselines_results.csv", index=False)
    df_by_plen.to_csv(results_dir / "baselines_by_prefix_len.csv", index=False)

    # Unseen context note
    notes = pd.DataFrame([
        {"model": "B1_Majority",
         "unseen_context_handling": "Always predicts the single most frequent train label, context-independent."},
        {"model": "B2_UnigramPrior",
         "unseen_context_handling": "Always predicts from singleton global prior, context-independent."},
        {"model": f"B3_Markov1_k{best_k}",
         "unseen_context_handling": f"If last technique not seen in train bigrams, falls back to singleton unigram prior. Add-k={best_k} smoothing otherwise."},
        {"model": f"B4_Markov2_m{best_m}",
         "unseen_context_handling": f"If bigram context has < {best_m} observations, backs off to 1st-order Markov. If that context is also unseen, falls back to singleton unigram prior."},
    ])
    notes.to_csv(results_dir / "baselines_unseen_context_notes.csv", index=False)

    print("\n" + "=" * 80)
    print("BASELINE RESULTS ON VALIDATION SET")
    print("=" * 80)
    print(df_overall.to_string(index=False))
    print("\n── By prefix length ──")
    print(df_by_plen.to_string(index=False))
    print(f"\nSaved baselines_results.csv, baselines_by_prefix_len.csv, "
          f"baselines_unseen_context_notes.csv")

    return {
        "df_overall": df_overall,
        "df_by_plen": df_by_plen,
        "best_k": best_k,
        "best_m": best_m,
    }


if __name__ == "__main__":
    run_baselines()
