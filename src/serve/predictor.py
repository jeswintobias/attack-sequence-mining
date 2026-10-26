"""Serving logic for the AttackCast.

Loads the artifacts once and answers the three questions the application asks:

- `predict_next`      what might come next, ranked, with the evidence behind it
- `localise_stage`    where in the ATT&CK kill chain the campaign appears to be
- `reconstruct_path`  which orderings of an observed set are most plausible

Every answer carries the model card's caveats and a `ContextInfo` describing how
much training support the prefix actually had, because a confident-looking
probability over an unseen context is this model's main failure mode.
"""

from __future__ import annotations

import math
import sys
from heapq import nlargest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models"),
          str(SRC / "mining"), str(SRC / "eval"), str(SRC / "serve")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from artifacts import Artifacts, load
from contract import (Candidate, ContextInfo, ModelInfo, OrderingCandidate,
                      PredictResponse, ReconstructResponse, RuleEvidence,
                      StageResponse, TacticPosition, TransitionEvidence)
from tactic_map import TACTIC_ORDER

CONTEXT_SEPARATOR = "\u0000"


def _is_subsequence(sub, seq) -> bool:
    iterator = iter(seq)
    return all(item in iterator for item in sub)


class AttackPathPredictor:
    """Stateless predictor over preloaded artifacts."""

    def __init__(self, artifacts: Artifacts | None = None):
        self.artifacts = load() if artifacts is None else artifacts
        self.vocabulary = self.artifacts.vocabulary
        self.names = self.artifacts.names
        self.tactics = self.artifacts.tactics
        self.markov = self.artifacts.markov
        self.add_k = float(self.markov.get("add_k", 0.001))
        self.backoff_threshold = float(self.markov.get("backoff_threshold", 5))
        self.label_counts = {k: float(v) for k, v in self.markov["label_counts"].items()}
        self.label_total = sum(self.label_counts.values()) or 1.0
        self.temperature = float(self.artifacts.calibration.get("temperature", 1.0) or 1.0)

        classes = np.asarray(self.artifacts.model.classes_, dtype=int)
        label_classes = np.asarray(self.artifacts.label_classes, dtype=object)
        self.class_names = label_classes[classes]
        self.known = set(self.vocabulary)

        # Rules indexed by their final antecedent item: a rule can only fire on a
        # prefix whose last technique matches, so this avoids scanning all of them.
        self.rules_by_anchor: dict[str, list[dict]] = {}
        for rule in self.artifacts.rules:
            anchor = rule["antecedent"][-1]
            self.rules_by_anchor.setdefault(anchor, []).append(rule)

        self._tactic_index = {name: i for i, name in enumerate(TACTIC_ORDER)}

    # ---- shared helpers ----------------------------------------------------
    def _model_info(self) -> ModelInfo:
        card = self.artifacts.card
        measured = card["evaluation"]["measured"]
        return ModelInfo(
            name=card["model"],
            trained_on=card["trained_on"],
            sequences=card["sequences"],
            examples=card["examples"],
            label_classes=card["label_classes"],
            metrics={k: round(float(v), 4) for k, v in measured.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)},
        )

    def _context_support(self, observed: list[str]) -> tuple[float, float]:
        first = sum(self.markov["order1"].get(observed[-1], {}).values())
        second = 0.0
        if len(observed) >= 2:
            key = CONTEXT_SEPARATOR.join((observed[-2], observed[-1]))
            second = sum(self.markov["order2"].get(key, {}).values())
        return float(first), float(second)

    def _context_info(self, observed: list[str], excluded: list[str]) -> ContextInfo:
        first, second = self._context_support(observed)
        if second >= self.backoff_threshold:
            backoff = "second_order"
        elif first > 0:
            backoff = "first_order"
        else:
            backoff = "prior"
        if backoff == "second_order" and second >= 20:
            confidence = "high"
        elif backoff in ("second_order", "first_order") and first >= 20:
            confidence = "medium"
        else:
            confidence = "low"
        return ContextInfo(
            observed_count=len(observed),
            source="learned_model" if len(observed) >= 2 else "first_order_fallback",
            confidence=confidence,
            backoff=backoff,
            second_order_support=second,
            first_order_support=first,
            excluded_already_observed=excluded,
            unknown_techniques=[t for t in observed if t not in self.known],
        )

    def _transition_evidence(self, observed: list[str], candidate: str) -> list[TransitionEvidence]:
        evidence = []
        counts1 = self.markov["order1"].get(observed[-1], {})
        total1 = sum(counts1.values())
        if total1:
            evidence.append(TransitionEvidence(
                context=[observed[-1]], order=1,
                count=float(counts1.get(candidate, 0.0)), context_total=float(total1),
                probability=float(counts1.get(candidate, 0.0) / total1),
            ))
        if len(observed) >= 2:
            key = CONTEXT_SEPARATOR.join((observed[-2], observed[-1]))
            counts2 = self.markov["order2"].get(key, {})
            total2 = sum(counts2.values())
            if total2:
                evidence.append(TransitionEvidence(
                    context=[observed[-2], observed[-1]], order=2,
                    count=float(counts2.get(candidate, 0.0)), context_total=float(total2),
                    probability=float(counts2.get(candidate, 0.0) / total2),
                ))
        return evidence

    def _fired_rules(self, observed: list[str]) -> dict[str, list[RuleEvidence]]:
        """Mined rules whose antecedent matches this prefix, keyed by consequent."""
        head = tuple(observed[:-1])
        fired: dict[str, list[RuleEvidence]] = {}
        for rule in self.rules_by_anchor.get(observed[-1], ()):
            antecedent = tuple(rule["antecedent"])
            if len(antecedent) > 1 and not _is_subsequence(antecedent[:-1], head):
                continue
            fired.setdefault(rule["consequent"], []).append(RuleEvidence(**rule))
        for consequent, items in fired.items():
            items.sort(key=lambda r: (-r.confidence, -r.support))
        return fired

    def _calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Temperature-scale the raw output (see artifacts.fit_calibration)."""
        if abs(self.temperature - 1.0) < 1e-6:
            return probabilities
        scaled = np.log(np.clip(probabilities, 1e-12, None)) / self.temperature
        scaled -= scaled.max()
        exponentiated = np.exp(scaled)
        return exponentiated / exponentiated.sum()

    def _probabilities(self, observed: list[str]) -> np.ndarray:
        """Calibrated class probabilities, or a first-order fallback.

        Every training prefix has length >= 2, so a single observed technique is
        out of distribution for the learned model: it returns confident nonsense
        there. Below two techniques we use the first-order transition
        distribution instead, and say so in `ContextInfo.source`.
        """
        from lightgbm_predictor import _make_features

        if len(observed) < 2:
            counts = self.markov["order1"].get(observed[-1], {})
            total = sum(counts.values())
            fallback = np.zeros(len(self.class_names), dtype=float)
            size = max(len(self.label_counts), 1)
            for index, name in enumerate(self.class_names):
                if total:
                    fallback[index] = ((counts.get(str(name), 0.0) + self.add_k)
                                       / (total + self.add_k * size))
                else:
                    fallback[index] = ((self.label_counts.get(str(name), 0.0) + self.add_k)
                                       / (self.label_total + self.add_k * size))
            return fallback / fallback.sum()

        frame = pd.DataFrame({"prefix": [tuple(observed)]})
        features = _make_features(frame, self.vocabulary)
        return self._calibrate(self.artifacts.model.predict_proba(features)[0])

    # ---- capability A: next-step prediction --------------------------------
    def predict_next(self, observed: list[str], top_k: int = 5) -> PredictResponse:
        probabilities = self._probabilities(observed)
        seen = set(observed)
        excluded = []
        if config.MASK_SEEN_TECHNIQUES:
            mask = np.array([name in seen for name in self.class_names])
            excluded = sorted(self.class_names[mask].tolist())
            probabilities = np.where(mask, 0.0, probabilities)
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total

        order = np.argsort(-probabilities, kind="stable")[:top_k]
        fired = self._fired_rules(observed)
        candidates = []
        for rank, index in enumerate(order, start=1):
            technique = str(self.class_names[index])
            candidates.append(Candidate(
                rank=rank,
                technique=technique,
                name=self.names.get(technique),
                probability=float(probabilities[index]),
                tactics=self.tactics.get(technique, []),
                transitions=self._transition_evidence(observed, technique),
                rules=fired.get(technique, []),
            ))
        return PredictResponse(
            candidates=candidates,
            context=self._context_info(observed, excluded),
            model=self._model_info(),
            caveats=self.artifacts.caveats,
        )

    # ---- capability B: stage localisation ----------------------------------
    def localise_stage(self, observed: list[str], top_k: int = 10) -> StageResponse:
        """Kill-chain position, from the tactics of observed and predicted techniques.

        Display only. Tactic features were measured not to improve prediction
        (DEC-20); this reports where the campaign appears to sit, it does not
        claim extra predictive power.
        """
        observed_tactics = set()
        unmapped = []
        for technique in observed:
            mapped = self.tactics.get(technique)
            if mapped:
                observed_tactics.update(mapped)
            else:
                unmapped.append(technique)

        prediction = self.predict_next(observed, top_k=top_k)
        weights = {name: 0.0 for name in TACTIC_ORDER}
        mass = 0.0
        for candidate in prediction.candidates:
            if not candidate.tactics:
                continue
            share = candidate.probability / max(len(candidate.tactics), 1)
            for tactic in candidate.tactics:
                if tactic in weights:
                    weights[tactic] += share
                    mass += share
        if mass > 0:
            weights = {k: v / mass for k, v in weights.items()}

        positions = [
            TacticPosition(tactic=name, index=index,
                           observed=name in observed_tactics,
                           predicted_weight=round(float(weights.get(name, 0.0)), 6))
            for index, name in enumerate(TACTIC_ORDER)
        ]
        reached = [p for p in positions if p.observed]
        return StageResponse(
            tactics=positions,
            furthest_observed=reached[-1].tactic if reached else None,
            unmapped_techniques=unmapped,
            caveats=self.artifacts.caveats + [
                "Kill-chain position is derived from the ATT&CK tactic taxonomy for "
                "display. Tactic features were measured not to improve prediction "
                "accuracy (DEC-20), so this is orientation, not evidence."
            ],
        )

    # ---- capability C: path reconstruction ---------------------------------
    def _log_transition(self, history: list[str], candidate: str) -> float:
        """Smoothed log P(candidate | history) with second to first order back-off."""
        vocabulary_size = max(len(self.label_counts), 1)
        if len(history) >= 2:
            key = CONTEXT_SEPARATOR.join((history[-2], history[-1]))
            counts = self.markov["order2"].get(key)
            if counts:
                total = sum(counts.values())
                if total >= self.backoff_threshold:
                    probability = ((counts.get(candidate, 0.0) + self.add_k)
                                   / (total + self.add_k * vocabulary_size))
                    return math.log(max(probability, 1e-12))
        if history:
            counts = self.markov["order1"].get(history[-1])
            if counts:
                total = sum(counts.values())
                probability = ((counts.get(candidate, 0.0) + self.add_k)
                               / (total + self.add_k * vocabulary_size))
                return math.log(max(probability, 1e-12))
        probability = (self.label_counts.get(candidate, 0.0) + self.add_k) / (
            self.label_total + self.add_k * vocabulary_size)
        return math.log(max(probability, 1e-12))

    def reconstruct_path(self, observed: list[str], top_n: int = 5,
                         beam_width: int = 50) -> ReconstructResponse:
        """Beam search over orderings of an observed technique set.

        There are k! orderings, so the search is bounded rather than exhaustive.
        `exhaustive` reports whether the beam was wide enough that nothing was
        pruned, in which case the ranking is over every permutation.
        """
        unique = list(dict.fromkeys(observed))
        beam: list[tuple[float, tuple[str, ...]]] = [(0.0, ())]
        considered = 0
        pruned = False
        for _ in range(len(unique)):
            extended: list[tuple[float, tuple[str, ...]]] = []
            for score, path in beam:
                remaining = [t for t in unique if t not in path]
                for candidate in remaining:
                    considered += 1
                    extended.append(
                        (score + self._log_transition(list(path), candidate),
                         path + (candidate,)))
            if len(extended) > beam_width:
                pruned = True
                beam = nlargest(beam_width, extended, key=lambda item: item[0])
            else:
                beam = extended

        best = nlargest(min(top_n, len(beam)), beam, key=lambda item: item[0])
        if best:
            top = best[0][0]
            weights = [math.exp(score - top) for score, _ in best]
            denominator = sum(weights) or 1.0
        else:
            weights, denominator = [], 1.0
        orderings = [
            OrderingCandidate(sequence=list(path), log_likelihood=round(score, 4),
                              relative_likelihood=round(weight / denominator, 6))
            for (score, path), weight in zip(best, weights)
        ]
        return ReconstructResponse(
            orderings=orderings,
            considered=considered,
            exhaustive=not pruned,
            caveats=self.artifacts.caveats + [
                "Orderings are scored with the second-order transition model and "
                "ranked by relative likelihood within the returned beam only, not "
                "over all k! permutations.",
                "Path reconstruction was not separately evaluated; unlike "
                "next-step prediction it carries no measured accuracy.",
            ],
        )

    # ---- catalogue for the technique picker --------------------------------
    def catalogue(self) -> list[dict]:
        """Every technique the model can accept or predict, with names and tactics."""
        return [
            {
                "technique": technique,
                "name": self.names.get(technique),
                "tactics": self.tactics.get(technique, []),
                "predictable": technique in set(self.class_names.tolist()),
                "train_frequency": round(
                    self.label_counts.get(technique, 0.0) / self.label_total, 6),
            }
            for technique in self.vocabulary
        ]
