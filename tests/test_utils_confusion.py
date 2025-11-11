# tests/test_utils_confusion.py
"""Basic tests for utils_confusion.py."""

import numpy as np
import pandas as pd

from confusion_matrix_explorer.utils_confusion import (
    ConfusionCounts,
    compute_confusion_counts,
    compute_metrics,
    format_counts_table,
    generate_synthetic_scores,
    safe_divide,
)


def test_generate_synthetic_scores_shape_and_labels():
    """Ensure generated synthetic data has expected structure."""
    df = generate_synthetic_scores(n_pos=5, n_neg=5, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"score", "label"}
    assert len(df) == 10


def test_compute_confusion_counts_known_case():
    """Test confusion count calculation on a simple deterministic example."""
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    threshold = 0.5

    counts = compute_confusion_counts(scores, labels, threshold)
    # Above threshold: indices 2, 3 => predicted positive
    # Actual positives: indices 2, 3 => both correct
    assert counts.tp == 2
    assert counts.fp == 0
    assert counts.fn == 0
    assert counts.tn == 2
    assert counts.total == 4


def test_safe_divide_behavior():
    """safe_divide should handle division and zero denominator correctly."""
    assert safe_divide(4, 2) == 2.0
    assert np.isnan(safe_divide(1, 0))


def test_compute_metrics_basic_values():
    """Test metric calculations from known confusion counts."""
    counts = ConfusionCounts(tp=50, fn=10, fp=5, tn=35)
    metrics = compute_metrics(counts)

    # Expected proportions
    assert np.isclose(metrics.sensitivity, 50 / (50 + 10))
    assert np.isclose(metrics.specificity, 35 / (35 + 5))
    assert np.isclose(metrics.precision, 50 / (50 + 5))
    assert np.isclose(metrics.npv, 35 / (35 + 10))
    assert np.isclose(metrics.accuracy, (50 + 35) / counts.total)
    assert np.isclose(metrics.prevalence, (50 + 10) / counts.total)


def test_format_counts_table_structure():
    """Check that confusion matrix table is correctly formatted."""
    counts = ConfusionCounts(tp=2, fn=3, fp=4, tn=5)
    table = format_counts_table(counts)

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["", "Disease Present", "Disease Absent"]
    assert table.iloc[0, 1] == 2  # TP
    assert table.iloc[0, 2] == 4  # FP
    assert table.iloc[1, 1] == 3  # FN
    assert table.iloc[1, 2] == 5  # TN
