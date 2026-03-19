from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lab3_probabilistic.app.classifier import ProbabilisticClassifier1D
from lab3_probabilistic.app.utils import gaussian_pdf


@dataclass
class PlotData:
    # All data required for plotting and metrics display
    xs: np.ndarray
    p1_pdf: np.ndarray
    p2_pdf: np.ndarray
    decisions: np.ndarray
    roots: list[float]
    false_alarm: float
    miss: float
    total_error: float


def _build_axis_range(
    mu1: float, s1: float, mu2: float, s2: float
) -> tuple[float, float]:
    # Build plotting range: cover both distributions (~±5 sigma)
    left = min(mu1 - 5 * s1, mu2 - 5 * s2)
    right = max(mu1 + 5 * s1, mu2 + 5 * s2)
    return left, right


def prepare_plot_data(
    mu1: float,
    s1: float,
    mu2: float,
    s2: float,
    p1: float,
    p2: float,
) -> PlotData:
    # Create classifier (contains decision rule + thresholds)
    clf = ProbabilisticClassifier1D(mu1, s1, mu2, s2, p1, p2)

    roots = clf.thresholds()

    # Generate x-axis range and grid
    left, right = _build_axis_range(mu1, s1, mu2, s2)
    xs = np.linspace(left, right, 2000)

    # Compute class-conditional PDFs
    p1_pdf = gaussian_pdf(xs, mu1, s1)
    p2_pdf = gaussian_pdf(xs, mu2, s2)

    # Compute decision label for each x (region classification)
    decisions = np.array([clf.decision_label(x) for x in xs])

    false_alarm, miss, total_error = clf.error_rates()

    return PlotData(
        xs=xs,
        p1_pdf=p1_pdf,
        p2_pdf=p2_pdf,
        decisions=decisions,
        roots=roots,
        false_alarm=false_alarm,
        miss=miss,
        total_error=total_error,
    )
