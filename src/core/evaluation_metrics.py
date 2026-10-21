"""Shared classification metrics for validation prediction outputs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def macro_recall_f1(y_true, y_pred, sample_weight=None):
    """Macro recall/F1 over labels present in validation targets.

    Labels absent from validation are excluded because their recall cannot be
    estimated from this split. A zero-division class receives a score of zero.
    """
    labels = np.unique(np.asarray(y_true, dtype=object))
    if not len(labels):
        return {"macro_recall": float("nan"), "macro_f1": float("nan")}
    _, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        sample_weight=sample_weight,
        zero_division=0,
    )
    return {"macro_recall": float(recall), "macro_f1": float(f1)}
