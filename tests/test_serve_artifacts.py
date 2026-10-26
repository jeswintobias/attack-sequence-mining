"""Tests for the serving layer.

Run from the repository root:
  python tests/test_serve_artifacts.py

The load-bearing test is `test_card_metric_matches_documentation`: it asserts
that the metric recorded in the shipped model card is the metric in the project
documentation. Together with the build step, which refuses to write artifacts
unless cross-validation reproduces that figure, this is what makes the
application's numbers defensible rather than merely plausible.

The remaining tests guard the behaviours the interface depends on: that masking
is applied, that probabilities are a usable distribution, that unseen contexts
are reported rather than hidden, and that the evidence attached to a prediction
is consistent with the transition tables it came from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"),
          str(SRC / "mining"), str(SRC / "eval"), str(SRC / "serve")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from artifacts import ARTIFACT_DIR, CV_TOLERANCE, EXPECTED_CV_MRR, load
from contract import PredictRequest, ReconstructRequest
from predictor import AttackPathPredictor

FAILURES: list[str] = []


def check(condition: bool, message: str):
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


# ---------------------------------------------------------------- artifacts
def test_artifacts_present():
    print("\nartifacts present and loadable")
    artifacts = load()
    check(artifacts.model is not None, "LightGBM model loads")
    check(len(artifacts.vocabulary) > 100, f"vocabulary loaded ({len(artifacts.vocabulary)})")
    check(len(artifacts.label_classes) > 90, f"label classes loaded ({len(artifacts.label_classes)})")
    check(len(artifacts.rules) > 0, f"PrefixSpan rules loaded ({len(artifacts.rules)})")
    check(len(artifacts.tactics) > 50, f"tactic map loaded ({len(artifacts.tactics)})")
    check(len(artifacts.names) > 200, f"technique names loaded ({len(artifacts.names)})")
    check(len(artifacts.caveats) >= 4, f"caveats present ({len(artifacts.caveats)})")
    return artifacts


def test_card_metric_matches_documentation(artifacts):
    """The shipped model must be the evaluated model."""
    print("\nmodel card metric matches the documented evaluation")
    evaluation = artifacts.card["evaluation"]
    measured = evaluation["measured"]["seq_mrr"]
    check(evaluation["expected_seq_mrr"] == EXPECTED_CV_MRR,
          f"card expectation pinned to {EXPECTED_CV_MRR}")
    check(abs(measured - EXPECTED_CV_MRR) <= CV_TOLERANCE,
          f"measured seq_mrr {measured:.4f} within {CV_TOLERANCE} of {EXPECTED_CV_MRR}")
    check(evaluation["measured"]["n_folds"] == config.N_CV_FOLDS,
          f"evaluated over {config.N_CV_FOLDS} folds")
    check(artifacts.card["test_data_loaded"] is False,
          "card records that the test split was not loaded")
    check(len(artifacts.card["known_negative_results"]) >= 4,
          "card carries the documented negative results")


def test_calibration_recorded(artifacts):
    """Over-confidence must be corrected before any number reaches a user."""
    print("\ncalibration is fitted and improves on the raw model")
    calibration = artifacts.calibration
    check("temperature" in calibration, "temperature recorded")
    check(calibration.get("ece_after", 1.0) < calibration.get("ece_before", 0.0),
          f"ECE improved: {calibration.get('ece_before')} -> {calibration.get('ece_after')}")
    check(calibration.get("ece_after", 1.0) < 0.05,
          f"post-calibration ECE under 0.05 ({calibration.get('ece_after')})")
    gap = abs(calibration.get("mean_confidence_after", 0.0)
              - calibration.get("observed_accuracy", 1.0))
    check(gap < 0.05,
          f"stated confidence tracks observed accuracy within 0.05 (gap {gap:.4f})")
    check(artifacts.card["calibration"]["temperature"] == calibration["temperature"],
          "model card and calibration artifact agree")


def test_single_technique_fallback(predictor):
    """One observed technique is out of distribution for the learned model."""
    print("\nsingle-technique input falls back to the transition model")
    response = predictor.predict_next(["T1059"], top_k=5)
    check(response.context.source == "first_order_fallback",
          "fallback path is used and declared")
    check(response.context.observed_count == 1, "observed count reported")
    for candidate in response.candidates:
        order1 = next((t for t in candidate.transitions if t.order == 1), None)
        if order1 is not None:
            check(abs(candidate.probability - order1.probability) < 0.01,
                  f"{candidate.technique} probability matches its empirical rate "
                  f"({candidate.probability:.3f} vs {order1.probability:.3f})")
    two = predictor.predict_next(["T1078", "T1059"], top_k=3)
    check(two.context.source == "learned_model",
          "two techniques use the learned model")


# ---------------------------------------------------------------- contract
def test_contract_normalisation():
    print("\nrequest validation and normalisation")
    request = PredictRequest(observed=["t1059", " T1078 ", "T1059.001"])
    check(request.observed == ["T1059", "T1078", "T1059"],
          "ids are upper-cased, trimmed, and subtechniques collapsed (DEC-01)")
    try:
        PredictRequest(observed=["powershell"])
        check(False, "non-ATT&CK input is rejected")
    except Exception:
        check(True, "non-ATT&CK input is rejected")
    try:
        ReconstructRequest(observed=["T1059"] * 9)
        check(False, "reconstruction input is capped at 8 techniques")
    except Exception:
        check(True, "reconstruction input is capped at 8 techniques")


# ---------------------------------------------------------------- prediction
def test_prediction(predictor):
    print("\nnext-step prediction")
    observed = ["T1078", "T1059"]
    response = predictor.predict_next(observed, top_k=5)
    check(len(response.candidates) == 5, "returns the requested number of candidates")
    check([c.rank for c in response.candidates] == [1, 2, 3, 4, 5], "ranks are 1..k")
    probabilities = [c.probability for c in response.candidates]
    check(probabilities == sorted(probabilities, reverse=True),
          "candidates are ordered by descending probability")
    check(all(0.0 <= p <= 1.0 for p in probabilities), "probabilities are in [0, 1]")
    check(sum(probabilities) <= 1.0 + 1e-6, "top-k probabilities do not exceed 1")
    check(not any(c.technique in set(observed) for c in response.candidates),
          "already-observed techniques are never predicted (DEC-17)")
    check(len(response.caveats) >= 4, "caveats travel with every prediction")
    check(response.model.metrics.get("seq_mrr") is not None,
          "response carries the evaluated metric")
    named = [c for c in response.candidates if c.name]
    check(len(named) >= 3, f"candidates carry human-readable names ({len(named)}/5)")
    return response


def test_evidence_consistency(predictor, response):
    print("\nevidence is consistent with the transition tables")
    top = response.candidates[0]
    for evidence in top.transitions:
        expected = evidence.count / evidence.context_total if evidence.context_total else 0.0
        check(abs(evidence.probability - expected) < 1e-6,
              f"order-{evidence.order} probability equals count/total")
        check(evidence.count <= evidence.context_total,
              f"order-{evidence.order} count does not exceed its context total")
    for candidate in response.candidates:
        for rule in candidate.rules:
            check(rule.consequent == candidate.technique,
                  "attached rules predict the candidate they are attached to")
            check(rule.antecedent[-1] == "T1059",
                  "fired rules anchor on the last observed technique")
    check(True, "evidence structure validated")


def test_unseen_context_is_reported(predictor):
    print("\nunseen context is surfaced, not hidden")
    response = predictor.predict_next(["T1659", "T1700"], top_k=3)
    check(response.context.backoff in ("second_order", "first_order", "prior"),
          f"back-off level reported ({response.context.backoff})")
    rare = predictor.predict_next(["T1078"], top_k=3)
    check(rare.context.second_order_support == 0.0,
          "a single observation has no second-order support")
    check(rare.context.backoff != "second_order",
          "a single observation does not claim second-order context")


# ---------------------------------------------------------------- stage
def test_stage_localisation(predictor):
    print("\nstage localisation")
    response = predictor.localise_stage(["T1078", "T1059"])
    check(len(response.tactics) == 14, f"all 14 tactics returned ({len(response.tactics)})")
    check([t.index for t in response.tactics] == list(range(14)),
          "tactics are in kill-chain order")
    check(any(t.observed for t in response.tactics), "observed tactics are flagged")
    total = sum(t.predicted_weight for t in response.tactics)
    check(abs(total - 1.0) < 1e-6 or total == 0.0,
          f"predicted weights form a distribution (sum={total:.4f})")
    check(any("DEC-20" in c for c in response.caveats),
          "stage view discloses that tactics do not improve accuracy")


# ---------------------------------------------------------------- paths
def test_path_reconstruction(predictor):
    print("\npath reconstruction")
    observed = ["T1078", "T1059", "T1082"]
    response = predictor.reconstruct_path(observed, top_n=3, beam_width=50)
    check(len(response.orderings) == 3, "returns the requested number of orderings")
    for ordering in response.orderings:
        check(sorted(ordering.sequence) == sorted(observed),
              "each ordering is a permutation of the input")
    likelihoods = [o.log_likelihood for o in response.orderings]
    check(likelihoods == sorted(likelihoods, reverse=True),
          "orderings are ranked by descending likelihood")
    check(response.exhaustive, "3 techniques with beam 50 covers all 6 permutations")
    narrow = predictor.reconstruct_path(["T1078", "T1059", "T1082", "T1105", "T1021"],
                                        top_n=2, beam_width=3)
    check(not narrow.exhaustive, "a narrow beam is reported as non-exhaustive")
    check(any("not separately evaluated" in c for c in response.caveats),
          "reconstruction discloses that it has no measured accuracy")


def test_catalogue(predictor):
    print("\ntechnique catalogue for the picker")
    catalogue = predictor.catalogue()
    check(len(catalogue) > 100, f"catalogue covers the vocabulary ({len(catalogue)})")
    named = [row for row in catalogue if row["name"]]
    check(len(named) / len(catalogue) > 0.9,
          f"over 90% have names ({len(named)}/{len(catalogue)})")
    check(any(row["predictable"] for row in catalogue),
          "predictable techniques are marked")
    powershell = [row for row in catalogue if row["technique"] == "T1059"]
    check(bool(powershell) and powershell[0]["name"] == "Command and Scripting Interpreter",
          "T1059 resolves to its ATT&CK name")


def main():
    print("=" * 70)
    print("SERVING LAYER TESTS")
    print("=" * 70)
    if not (ARTIFACT_DIR / "model_card.json").exists():
        print(f"\nNo artifacts in {ARTIFACT_DIR}.")
        print("Run: python src/serve/artifacts.py")
        sys.exit(1)

    artifacts = test_artifacts_present()
    test_card_metric_matches_documentation(artifacts)
    test_calibration_recorded(artifacts)
    test_contract_normalisation()
    predictor = AttackPathPredictor(artifacts)
    response = test_prediction(predictor)
    test_evidence_consistency(predictor, response)
    test_unseen_context_is_reported(predictor)
    test_single_technique_fallback(predictor)
    test_stage_localisation(predictor)
    test_path_reconstruction(predictor)
    test_catalogue(predictor)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for message in FAILURES:
            print(f"  - {message}")
        sys.exit(1)
    print("ALL SERVING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
