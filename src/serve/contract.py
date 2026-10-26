"""Request and response schemas for the AttackCast API.

The contract is deliberately shaped around what the evaluation supports. Three
choices are worth stating because they are not accidental:

- `candidates` is always a ranked **list**. Top-1 accuracy is 0.369 while top-5
  is 0.756, and a reused two-technique context is followed by 6.02 distinct
  techniques on average, so a single-answer response would misrepresent the
  model. Clients are given the distribution, not a verdict.
- Every prediction carries `context`, which reports how much training support
  the observed prefix actually had. A confident-looking probability over an
  unseen context is the main way this model can mislead, so the back-off level
  travels with the answer rather than being buried.
- Every prediction carries `caveats`, populated from the model card. The
  reported-order limitation has to reach the interface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

TECHNIQUE_PATTERN = r"^T\d{4}$"


class RuleEvidence(BaseModel):
    """A mined sequential rule that fired for this prefix."""

    antecedent: list[str] = Field(description="Ordered techniques forming the rule body")
    consequent: str = Field(description="Technique the rule predicts")
    confidence: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=0, description="Duplicate-weighted support in training")


class TransitionEvidence(BaseModel):
    """Observed transition counts behind a candidate, for explanation."""

    context: list[str] = Field(description="Context the count was measured over")
    order: Literal[1, 2] = Field(description="Markov order of this evidence")
    count: float = Field(ge=0.0, description="Duplicate-weighted observations")
    context_total: float = Field(ge=0.0, description="Total observations for the context")
    probability: float = Field(ge=0.0, le=1.0)


class Candidate(BaseModel):
    """One possible next technique, with the evidence supporting it."""

    rank: int = Field(ge=1)
    technique: str = Field(pattern=TECHNIQUE_PATTERN)
    name: str | None = Field(default=None, description="None when ATT&CK has no name on record")
    probability: float = Field(ge=0.0, le=1.0, description="Model probability, masked and renormalised")
    tactics: list[str] = Field(default_factory=list)
    transitions: list[TransitionEvidence] = Field(default_factory=list)
    rules: list[RuleEvidence] = Field(default_factory=list)


class ContextInfo(BaseModel):
    """How much training support the observed prefix had."""

    observed_count: int = Field(ge=0)
    source: Literal["learned_model", "first_order_fallback"] = Field(
        default="learned_model",
        description="first_order_fallback when fewer than two techniques were "
                    "observed, since every training prefix has length >= 2",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="low",
        description="Derived from training support for the context, not from the "
                    "probability value, which is only coarsely calibrated",
    )
    backoff: Literal["second_order", "first_order", "prior"] = Field(
        description="Deepest Markov context with usable support"
    )
    second_order_support: float = Field(ge=0.0)
    first_order_support: float = Field(ge=0.0)
    excluded_already_observed: list[str] = Field(
        default_factory=list,
        description="Candidates removed because the prefix already contains them",
    )
    unknown_techniques: list[str] = Field(
        default_factory=list,
        description="Submitted techniques absent from the training vocabulary",
    )


class PredictRequest(BaseModel):
    observed: list[str] = Field(min_length=1, max_length=32,
                                description="Techniques in the order they were observed")
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("observed")
    @classmethod
    def _normalise(cls, value: list[str]) -> list[str]:
        cleaned = []
        for item in value:
            token = str(item).strip().upper().split(".")[0]   # collapse subtechniques, DEC-01
            if not token.startswith("T") or not token[1:].isdigit():
                raise ValueError(f"Not an ATT&CK technique id: {item!r}")
            cleaned.append(token)
        return cleaned


class ModelInfo(BaseModel):
    name: str
    trained_on: str
    sequences: int
    examples: int
    label_classes: int
    metrics: dict[str, float]


class PredictResponse(BaseModel):
    candidates: list[Candidate]
    context: ContextInfo
    model: ModelInfo
    caveats: list[str]


class TacticPosition(BaseModel):
    tactic: str
    index: int = Field(ge=0, description="Position in kill-chain order")
    observed: bool = Field(description="A technique in the prefix maps to this tactic")
    predicted_weight: float = Field(
        ge=0.0, le=1.0,
        description="Share of predicted probability mass whose technique maps here",
    )


class StageRequest(BaseModel):
    observed: list[str] = Field(min_length=1, max_length=32)

    _normalise = field_validator("observed")(PredictRequest._normalise.__func__)


class StageResponse(BaseModel):
    tactics: list[TacticPosition]
    furthest_observed: str | None = Field(
        default=None, description="Latest kill-chain tactic the prefix reaches"
    )
    unmapped_techniques: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class OrderingCandidate(BaseModel):
    sequence: list[str]
    log_likelihood: float = Field(description="Sum of log transition probabilities")
    relative_likelihood: float = Field(
        ge=0.0, le=1.0, description="Normalised over the returned beam, not over all orderings"
    )


class ReconstructRequest(BaseModel):
    observed: list[str] = Field(min_length=2, max_length=8,
                                description="Unordered technique set; capped at 8, see caveats")
    top_n: int = Field(default=5, ge=1, le=20)
    beam_width: int = Field(default=50, ge=1, le=500)

    _normalise = field_validator("observed")(PredictRequest._normalise.__func__)


class ReconstructResponse(BaseModel):
    orderings: list[OrderingCandidate]
    considered: int = Field(description="Orderings scored by the beam search")
    exhaustive: bool = Field(description="True when the beam covered every permutation")
    caveats: list[str] = Field(default_factory=list)
