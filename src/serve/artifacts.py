"""Build and load the serving artifacts for the AttackCast.

Run from the repository root:
  python src/serve/artifacts.py            # build into artifacts/
  python src/serve/artifacts.py --verify   # load and re-check the model card

Why a build step exists
-----------------------
Until now every script fitted a model in memory and discarded it, so there was
no object that could be served and no guarantee that a served model matched an
evaluated one. The build does two things in order:

1. **Re-derives the headline metric** by running the documented 5-fold grouped
   cross-validation, and refuses to continue if it does not reproduce the figure
   in `docs/ACCURACY_INVESTIGATION.md`. This is what makes the application's
   numbers defensible: the configuration being shipped is provably the
   configuration that was measured.
2. **Refits that same configuration on the whole development set** (2,178
   sequences) and persists it alongside the Markov tables, the mined PrefixSpan
   rules, the tactic map, the technique names and a model card.

The sealed test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"),
          str(SRC / "mining"), str(SRC / "eval"), str(SRC / "serve")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config

ARTIFACT_DIR = config.BASE_DIR / "artifacts"

# The figure recorded in docs/ACCURACY_INVESTIGATION.md section 13 and DEC-18.
EXPECTED_CV_MRR = 0.5386
CV_TOLERANCE = 0.0015

LGB_PARAMS = {"num_leaves": 31, "min_child_samples": 10, "reg_lambda": 1.0}
LGB_ROUNDS = 101
MARKOV_K = 0.001
MARKOV_M = 5
PREFIXSPAN_SUPPORT = 0.01
PREFIXSPAN_CONF = 0.7
PREFIXSPAN_MAX_LEN = 4

CAVEATS = [
    "Predictions rank the next technique in *reported* order, which is not "
    "established to be execution order (DEC-05: ordering is non-random at "
    "p=0.005, but non-random is not chronological).",
    "Several next techniques are usually plausible: a reused two-technique "
    "context is followed by 6.02 distinct techniques on average. Treat the "
    "ranked list as a shortlist, not a verdict.",
    "Top-1 accuracy is 0.369; top-5 is 0.756. Read the top five, not the top one.",
    "Techniques already present in the prefix are excluded, because "
    "first-occurrence deduplication (DEC-03) makes them impossible targets. "
    "This is a property of the training data, not of real attacker behaviour.",
    "Trained on 2,178 incident sequences from the CTID sightings data. Not "
    "validated on any live enterprise environment.",
    "The raw model is over-confident; probabilities are temperature-scaled on "
    "out-of-fold predictions before display. Ranking is what the evaluation "
    "measured, so prefer rank and the observed transition counts over the "
    "probability when judging a candidate.",
    "At least two observed techniques are required for the learned model. A "
    "single technique falls back to the first-order transition model, because "
    "every training prefix has length 2 or more.",
]


def _lightgbm():
    import lightgbm as lgb
    from lightgbm_predictor import _base_params

    params = _base_params(dict(LGB_PARAMS), LGB_ROUNDS)
    params["n_jobs"] = 1
    return lgb.LGBMClassifier(**params)


def rederive_cv_metric(examples: pd.DataFrame) -> dict:
    """Re-run the documented cross-validation and return its metrics."""
    from adapters import SklearnProbaAdapter
    from cv_harness import run_cv

    folds, _, pooled = run_cv(
        lambda: SklearnProbaAdapter(_lightgbm, "LightGBM_softprob"),
        "LightGBM_softprob", examples, verbose=False,
    )
    return {
        "seq_mrr": float(pooled["seq_mrr"]),
        "seq_top_1": float(pooled["seq_top_1"]),
        "seq_top_3": float(pooled["seq_top_3"]),
        "seq_top_5": float(pooled["seq_top_5"]),
        "seq_macro_recall": float(pooled["seq_macro_recall"]),
        "seq_macro_f1": float(pooled["seq_macro_f1"]),
        "seq_mrr_sd_across_folds": float(pooled["seq_mrr_sd"]),
        "n_folds": int(pooled["n_folds"]),
        "per_fold_seq_mrr": [round(v, 4) for v in folds["seq_mrr"].tolist()],
    }


def _fit_markov(examples: pd.DataFrame) -> dict:
    """Transition tables, stored as plain dicts so loading needs no model classes."""
    from collections import Counter, defaultdict

    trans1, trans2 = defaultdict(Counter), defaultdict(Counter)
    labels = Counter()
    for prefix, nxt, dup in zip(examples["prefix"], examples["next"].astype(str),
                                examples["dup_count"].to_numpy(dtype=float)):
        trans1[prefix[-1]][nxt] += dup
        if len(prefix) >= 2:
            trans2[(prefix[-2], prefix[-1])][nxt] += dup
        labels[nxt] += dup
    return {
        "order1": {context: dict(counts) for context, counts in trans1.items()},
        "order2": {"\u0000".join(context): dict(counts)
                   for context, counts in trans2.items()},
        "label_counts": dict(labels),
        "add_k": MARKOV_K,
        "backoff_threshold": MARKOV_M,
    }


def _mine_rules(examples: pd.DataFrame) -> list[dict]:
    """PrefixSpan sequential rules at the B5 configuration (DEC-10)."""
    from prefixspan import prefixspan_mine

    unique = pd.DataFrame({
        "full_seq": examples["full_seq"].map(tuple),
        "dup_count": examples["dup_count"].to_numpy(),
    }).drop_duplicates("full_seq")
    sequences = unique["full_seq"].tolist()
    weights = unique["dup_count"].astype(int).tolist()
    total = sum(weights)
    patterns = prefixspan_mine(
        sequences, min_support=int(total * PREFIXSPAN_SUPPORT),
        max_len=PREFIXSPAN_MAX_LEN, weights=weights,
    )
    support = {tuple(pattern): value for value, pattern in patterns}
    rules = []
    for pattern, value in support.items():
        if len(pattern) < 2:
            continue
        antecedent, consequent = tuple(pattern[:-1]), pattern[-1]
        confidence = value / support.get(antecedent, total)
        if confidence >= PREFIXSPAN_CONF:
            rules.append({
                "antecedent": list(antecedent),
                "consequent": consequent,
                "confidence": round(float(confidence), 6),
                "support": int(value),
            })
    return sorted(rules, key=lambda r: (-r["confidence"], -r["support"]))


def _expected_calibration_error(probabilities: np.ndarray, correct: np.ndarray,
                                bins: int = 10) -> float:
    """Gap between stated confidence and observed accuracy, averaged over bins."""
    confidence = probabilities.max(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def fit_calibration(examples: pd.DataFrame) -> dict:
    """Temperature scaling fitted on out-of-fold predictions.

    The raw model is badly over-confident: on a single-technique context it can
    state 90.6% for a candidate whose observed transition rate is 8.8%. The
    ranking is what the evaluation measured; calibration was never tested, and
    displaying an uncalibrated number to an analyst would be misleading. A single
    temperature is fitted on out-of-fold probabilities, so it is not tuned on
    data the final model also fitted.
    """
    from scipy.optimize import minimize_scalar
    from sklearn.preprocessing import LabelEncoder

    from cv import development_folds
    from lightgbm_predictor import _make_features, _sequence_weights

    # Each fold trains on a different label subset, so fold probability vectors
    # have different widths. Scatter them onto the global label space, leaving
    # zero where a fold never saw a class -- which is what that model implies.
    global_labels = sorted(set(examples["next"].astype(str)))
    global_index = {label: i for i, label in enumerate(global_labels)}

    collected, truths = [], []
    for _, fit_idx, score_idx in development_folds(examples):
        fit = examples.iloc[fit_idx].reset_index(drop=True)
        score = examples.iloc[score_idx].reset_index(drop=True)
        encoder = LabelEncoder().fit(fit["next"].astype(str))
        vocabulary = sorted(set(fit["next"].astype(str))
                            | {t for p in fit["prefix"] for t in p})
        model = _lightgbm()
        model.fit(_make_features(fit, vocabulary),
                  encoder.transform(fit["next"].astype(str)),
                  sample_weight=_sequence_weights(fit))
        probabilities = model.predict_proba(_make_features(score, vocabulary))
        names = np.asarray(encoder.classes_, dtype=object)[
            np.asarray(model.classes_, dtype=int)]
        columns = np.array([global_index[name] for name in names], dtype=int)
        for row, target in enumerate(score["next"].astype(str)):
            scattered = np.zeros(len(global_labels), dtype=float)
            scattered[columns] = probabilities[row]
            collected.append(scattered)
            truths.append(global_index.get(target, -1))

    probabilities = np.vstack(collected)
    truths = np.asarray(truths)
    usable = truths >= 0
    probabilities, truths = probabilities[usable], truths[usable]
    log_probabilities = np.log(np.clip(probabilities, 1e-12, None))

    def negative_log_likelihood(temperature: float) -> float:
        scaled = log_probabilities / max(temperature, 1e-3)
        scaled -= scaled.max(axis=1, keepdims=True)
        normalised = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return float(-normalised[np.arange(len(truths)), truths].mean())

    result = minimize_scalar(negative_log_likelihood, bounds=(0.05, 20.0),
                             method="bounded")
    temperature = float(result.x)

    def apply(temp: float) -> np.ndarray:
        scaled = log_probabilities / temp
        scaled -= scaled.max(axis=1, keepdims=True)
        exponentiated = np.exp(scaled)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    correct = (probabilities.argmax(axis=1) == truths).astype(float)
    return {
        "method": "temperature scaling on out-of-fold log-probabilities",
        "temperature": round(temperature, 4),
        "nll_before": round(negative_log_likelihood(1.0), 4),
        "nll_after": round(negative_log_likelihood(temperature), 4),
        "ece_before": round(_expected_calibration_error(probabilities, correct), 4),
        "ece_after": round(_expected_calibration_error(apply(temperature), correct), 4),
        "mean_confidence_before": round(float(probabilities.max(axis=1).mean()), 4),
        "mean_confidence_after": round(float(apply(temperature).max(axis=1).mean()), 4),
        "observed_accuracy": round(float(correct.mean()), 4),
        "n_out_of_fold_examples": int(len(truths)),
    }


def build(out_dir: Path = ARTIFACT_DIR, skip_cv: bool = False) -> dict:
    from sklearn.preprocessing import LabelEncoder

    from cv import assert_test_sealed, load_development_examples
    from lightgbm_predictor import _make_features, _sequence_weights
    from tactic_map import load_mapping
    from build_technique_names import load_names

    assert_test_sealed()
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_development_examples()
    print(f"Development set: {len(examples):,} examples / "
          f"{examples['seq_id'].nunique():,} sequences. Test split not loaded.")

    if skip_cv:
        print("\n[1/4] SKIPPING cross-validation re-derivation (--skip-cv)")
        metrics = {"seq_mrr": float("nan"), "skipped": True}
    else:
        print(f"\n[1/3] Re-deriving the documented {config.N_CV_FOLDS}-fold metric...")
        metrics = rederive_cv_metric(examples)
        delta = abs(metrics["seq_mrr"] - EXPECTED_CV_MRR)
        print(f"      seq_mrr = {metrics['seq_mrr']:.4f}  "
              f"(expected {EXPECTED_CV_MRR:.4f}, delta {delta:.4f})")
        if delta > CV_TOLERANCE:
            raise RuntimeError(
                f"Cross-validation does not reproduce the documented metric: got "
                f"{metrics['seq_mrr']:.4f}, expected {EXPECTED_CV_MRR:.4f} "
                f"(tolerance {CV_TOLERANCE}). The serving configuration has "
                f"drifted from the evaluated one; do not ship this build."
            )
        print("      reproduced within tolerance: this configuration is the evaluated one")

    print("\n[2/4] Fitting on the full development set and mining rules...")
    encoder = LabelEncoder().fit(examples["next"].astype(str))
    y = encoder.transform(examples["next"].astype(str))
    vocabulary = sorted(set(examples["next"].astype(str))
                        | {t for p in examples["prefix"] for t in p})
    model = _lightgbm()
    model.fit(_make_features(examples, vocabulary), y,
              sample_weight=_sequence_weights(examples))
    markov = _fit_markov(examples)
    rules = _mine_rules(examples)
    tactics = {technique: sorted(values) for technique, values in load_mapping().items()}
    names = load_names()
    print(f"      model classes: {len(encoder.classes_)}  "
          f"features: {len(vocabulary) * 5 + 1}")
    print(f"      markov contexts: {len(markov['order1'])} first-order, "
          f"{len(markov['order2'])} second-order")
    print(f"      prefixspan rules at sup={PREFIXSPAN_SUPPORT} "
          f"conf>={PREFIXSPAN_CONF}: {len(rules)}")

    if skip_cv:
        calibration = {"method": "not fitted (--skip-cv)", "temperature": 1.0}
    else:
        print("\n[3/4] Fitting calibration on out-of-fold predictions...")
        calibration = fit_calibration(examples)
        print(f"      temperature = {calibration['temperature']:.3f}")
        print(f"      mean confidence {calibration['mean_confidence_before']:.3f} -> "
              f"{calibration['mean_confidence_after']:.3f} "
              f"(observed accuracy {calibration['observed_accuracy']:.3f})")
        print(f"      ECE {calibration['ece_before']:.4f} -> {calibration['ece_after']:.4f}")

    print("\n[4/4] Writing artifacts...")
    joblib.dump(model, out_dir / "lightgbm_model.joblib", compress=3)
    joblib.dump(markov, out_dir / "markov.joblib", compress=3)
    (out_dir / "vocabulary.json").write_text(json.dumps(vocabulary, indent=1))
    (out_dir / "label_classes.json").write_text(
        json.dumps([str(c) for c in encoder.classes_], indent=1))
    (out_dir / "prefixspan_rules.json").write_text(json.dumps(rules, indent=1))
    (out_dir / "tactic_map.json").write_text(json.dumps(tactics, indent=1))
    (out_dir / "technique_names.json").write_text(json.dumps(names, indent=1))
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=1))

    card = {
        "application": "AttackCast",
        "tagline": "where the intrusion goes next, ranked by evidence",
        "model": "LightGBM multiclass softprob over prefix-presence features",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": "sightings_v2 development set (original train + val splits)",
        "sequences": int(examples["seq_id"].nunique()),
        "examples": int(len(examples)),
        "label_classes": int(len(encoder.classes_)),
        "feature_count": int(len(vocabulary) * 5 + 1),
        "hyperparameters": {**LGB_PARAMS, "n_estimators": LGB_ROUNDS},
        "evaluation": {
            "protocol": f"{config.N_CV_FOLDS}-fold StratifiedGroupKFold over the "
                        "development set, grouped by sequence, stratified by length "
                        "(DEC-18); metrics are sequence-balanced",
            "expected_seq_mrr": EXPECTED_CV_MRR,
            "measured": metrics,
            "paired_reference": {
                "vs_markov2_seq_mrr_diff": -0.0898,
                "ci": [-0.0967, -0.0830],
                "note": "Markov-2 minus LightGBM; LightGBM is ahead and the CI excludes zero",
            },
        },
        "calibration": calibration,
        "masking": {
            "enabled": bool(config.MASK_SEEN_TECHNIQUES),
            "reason": "DEC-17: first-occurrence deduplication makes an already "
                      "observed technique an impossible target",
        },
        "components": {
            "ranking": "lightgbm_model.joblib",
            "explanation_transitions": "markov.joblib",
            "explanation_rules": "prefixspan_rules.json",
            "stage_display": "tactic_map.json",
        },
        "known_negative_results": [
            "Mined pattern features add nothing to the ranking (-0.0059); DEC-19",
            "Tactic abstraction does not improve ranking (+0.0013, not replicated); DEC-20",
            "Graph/Node2Vec, tactic and interaction features add +0.0012, CI straddles zero; DEC-21",
            "Fixed-weight ensembling with Markov-2 gains +0.0005; DEC-19",
        ],
        "caveats": CAVEATS,
        "test_data_loaded": False,
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")

    for path in sorted(out_dir.iterdir()):
        print(f"      {path.name:<26} {path.stat().st_size / 1024:>9.1f} KB")
    print(f"\nArtifacts written to {out_dir}")
    return card


@dataclass
class Artifacts:
    """Everything the predictor needs, loaded once at startup."""

    model: object
    vocabulary: list[str]
    label_classes: list[str]
    markov: dict
    rules: list[dict]
    tactics: dict[str, list[str]]
    names: dict[str, str]
    calibration: dict
    card: dict

    @property
    def caveats(self) -> list[str]:
        return list(self.card.get("caveats", []))


def load(out_dir: Path = ARTIFACT_DIR) -> Artifacts:
    if not (out_dir / "model_card.json").exists():
        raise FileNotFoundError(
            f"No artifacts in {out_dir}. Run: python src/serve/artifacts.py"
        )
    return Artifacts(
        model=joblib.load(out_dir / "lightgbm_model.joblib"),
        vocabulary=json.loads((out_dir / "vocabulary.json").read_text()),
        label_classes=json.loads((out_dir / "label_classes.json").read_text()),
        markov=joblib.load(out_dir / "markov.joblib"),
        rules=json.loads((out_dir / "prefixspan_rules.json").read_text()),
        tactics=json.loads((out_dir / "tactic_map.json").read_text()),
        names=json.loads((out_dir / "technique_names.json").read_text()),
        calibration=json.loads((out_dir / "calibration.json").read_text())
        if (out_dir / "calibration.json").exists() else {"temperature": 1.0},
        card=json.loads((out_dir / "model_card.json").read_text()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="load existing artifacts and print the model card summary")
    parser.add_argument("--skip-cv", action="store_true",
                        help="skip the metric re-derivation (development only; never ship)")
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()

    if args.verify:
        artifacts = load(args.out)
        card = artifacts.card
        print(f"model:     {card['model']}")
        print(f"built:     {card['built_at']}")
        print(f"trained:   {card['sequences']:,} sequences / {card['examples']:,} examples "
              f"/ {card['label_classes']} classes")
        measured = card["evaluation"]["measured"]
        print(f"seq MRR:   {measured.get('seq_mrr'):.4f} "
              f"(expected {card['evaluation']['expected_seq_mrr']})")
        print(f"top-1/3/5: {measured.get('seq_top_1'):.4f} / "
              f"{measured.get('seq_top_3'):.4f} / {measured.get('seq_top_5'):.4f}")
        print(f"rules:     {len(artifacts.rules)}  tactics: {len(artifacts.tactics)}  "
              f"names: {len(artifacts.names)}")
        print(f"caveats:   {len(artifacts.caveats)}")
        return
    build(args.out, skip_cv=args.skip_cv)


if __name__ == "__main__":
    main()
