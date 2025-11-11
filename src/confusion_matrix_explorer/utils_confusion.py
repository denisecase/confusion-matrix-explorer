# utils_confusion.py
"""Utility functions and classes for confusion matrix analysis.

This module provides:
- ConfusionCounts: dataclass for storing 2x2 confusion matrix counts
- ConfusionMetrics: dataclass for storing computed diagnostic metrics
- generate_synthetic_scores: create synthetic datasets for testing
- compute_confusion_counts: calculate TP, FP, TN, FN from scores and labels
- compute_metrics: calculate standard diagnostic metrics from counts
- format_counts_table: format confusion matrix for display
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfusionCounts:
    """Counts for a 2x2 confusion matrix.

    Attributes:
    tp : int
        True positives - correctly predicted positive cases.
    fn : int
        False negatives - positive cases incorrectly predicted as negative.
    fp : int
        False positives - negative cases incorrectly predicted as positive.
    tn : int
        True negatives - correctly predicted negative cases.
    """

    tp: int
    fn: int
    fp: int
    tn: int

    @property
    def total(self) -> int:
        """Return the total count of all cases in the confusion matrix.

        Returns:
        int
            Sum of true positives, false negatives, false positives, and true negatives.
        """
        return self.tp + self.fn + self.fp + self.tn


@dataclass
class ConfusionMetrics:
    """Computed diagnostic metrics from confusion matrix counts.

    Attributes:
    sensitivity : float
        True positive rate (recall) - proportion of actual positives correctly identified.
    specificity : float
        True negative rate - proportion of actual negatives correctly identified.
    precision : float
        Positive predictive value - proportion of predicted positives that are actual positives.
    npv : float
        Negative predictive value - proportion of predicted negatives that are actual negatives.
    accuracy : float
        Overall proportion of correct predictions.
    prevalence : float
        Proportion of actual positive cases in the population.
    """

    sensitivity: float
    specificity: float
    precision: float
    npv: float
    accuracy: float
    prevalence: float


def generate_synthetic_scores(
    n_pos: int = 200,
    n_neg: int = 200,
    pos_mean: float = 1.5,
    neg_mean: float = 0.0,
    pos_std: float = 1.0,
    neg_std: float = 1.0,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Generate a simple synthetic dataset of scores.

    - disease present (label = 1)
    - disease absent (label = 0)

    Returns a DataFrame with columns:
    - score: float
    - label: 0 or 1
    """
    rng = np.random.default_rng(random_state)

    pos_scores = rng.normal(loc=pos_mean, scale=pos_std, size=n_pos)
    neg_scores = rng.normal(loc=neg_mean, scale=neg_std, size=n_neg)

    df_pos = pd.DataFrame({"score": pos_scores, "label": 1})
    df_neg = pd.DataFrame({"score": neg_scores, "label": 0})

    return pd.concat([df_pos, df_neg], ignore_index=True)


def compute_confusion_counts(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> ConfusionCounts:
    """Compute TP, FP, TN, FN given scores, labels, and a decision threshold.

    By convention here:
    - score >= threshold -> predicted positive
    - score < threshold  -> predicted negative
    """
    predicted_positive = scores >= threshold
    predicted_negative = scores < threshold

    actual_positive = labels == 1
    actual_negative = labels == 0

    tp = int(np.sum(predicted_positive & actual_positive))
    fn = int(np.sum(predicted_negative & actual_positive))
    fp = int(np.sum(predicted_positive & actual_negative))
    tn = int(np.sum(predicted_negative & actual_negative))

    return ConfusionCounts(tp=tp, fn=fn, fp=fp, tn=tn)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or float('nan') if denominator is zero."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def compute_metrics(counts: ConfusionCounts) -> ConfusionMetrics:
    """Compute standard diagnostic metrics from confusion counts.

    Returns values as proportions (0.0 to 1.0).
    You can convert to percentages in the app.
    """
    tp = counts.tp
    fn = counts.fn
    fp = counts.fp
    tn = counts.tn
    total = counts.total

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    accuracy = safe_divide(tp + tn, total)
    prevalence = safe_divide(tp + fn, total)

    return ConfusionMetrics(
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        npv=npv,
        accuracy=accuracy,
        prevalence=prevalence,
    )


def format_counts_table(counts: ConfusionCounts) -> pd.DataFrame:
    """Build a 2x2 confusion matrix table for display.

                Disease Present   Disease Absent
    Test +
    Test -
    """
    return pd.DataFrame(
        {
            "": ["Test Positive", "Test Negative"],
            "Disease Present": [counts.tp, counts.fn],
            "Disease Absent": [counts.fp, counts.tn],
        }
    )
