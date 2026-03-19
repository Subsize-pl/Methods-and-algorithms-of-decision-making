import math
from typing import Tuple, List
import numpy as np

from lab3_probabilistic.app.utils import gaussian_cdf, decision_boundary_roots


class ProbabilisticClassifier1D:
    """
    Minimal 1D probabilistic classifier for two Gaussian classes.

    Usage:
      - build instance with parameters (mu1,s1), (mu2,s2) and priors p1,p2
      - call decision_regions() to get threshold points
      - call error_rates() to compute false alarm / miss / total error
    """

    def __init__(
        self,
        mu1: float,
        s1: float,
        mu2: float,
        s2: float,
        p1: float = 0.5,
        p2: float = 0.5,
    ):
        self.mu1 = float(mu1)
        self.s1 = float(s1)
        self.mu2 = float(mu2)
        self.s2 = float(s2)
        self.p1 = float(p1)
        self.p2 = float(p2)
        if abs(self.p1 + self.p2 - 1.0) > 1e-9:
            raise ValueError("Priors must sum to 1.0")

    def thresholds(self) -> List[float]:
        """Return decision boundary points (0,1 or 2 real roots)."""
        return decision_boundary_roots(
            self.mu1, self.s1, self.mu2, self.s2, self.p1, self.p2
        )

    def decision_label(self, x: float) -> int:
        """Decision at a scalar x: returns 1 or 2 (class indices)."""
        p1x = (
            (1.0 / (math.sqrt(2 * math.pi) * self.s1))
            * math.exp(-0.5 * ((x - self.mu1) / self.s1) ** 2)
            * self.p1
        )
        p2x = (
            (1.0 / (math.sqrt(2 * math.pi) * self.s2))
            * math.exp(-0.5 * ((x - self.mu2) / self.s2) ** 2)
            * self.p2
        )
        return 1 if p1x >= p2x else 2

    def error_rates(self) -> Tuple[float, float, float]:
        """Compute classification errors using analytic integrals (via CDF)."""

        # points where decision changes (boundaries between classes)
        roots = self.thresholds()

        # approximate "infinity" range (covers almost all probability mass)
        left = min(self.mu1 - 6 * self.s1, self.mu2 - 6 * self.s2)
        right = max(self.mu1 + 6 * self.s1, self.mu2 + 6 * self.s2)

        # split real line into intervals using roots
        pts = [left] + roots + [right]

        false_alarm = 0.0  # P(decide 1 | true 2)
        miss = 0.0  # P(decide 2 | true 1)

        # iterate over all intervals
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]

            # take midpoint to determine decision on this interval
            m = 0.5 * (a + b)
            label = self.decision_label(m)

            if label == 1:
                # region classified as class 1
                # error here = how much class 2 falls into this region
                false_alarm += self.p2 * (
                    gaussian_cdf(np.array([b]), self.mu2, self.s2)[0]
                    - gaussian_cdf(np.array([a]), self.mu2, self.s2)[0]
                )
            else:
                # region classified as class 2
                # error here = how much class 1 falls into this region
                miss += self.p1 * (
                    gaussian_cdf(np.array([b]), self.mu1, self.s1)[0]
                    - gaussian_cdf(np.array([a]), self.mu1, self.s1)[0]
                )

        # weighted total error (using priors)
        total_error = miss + false_alarm

        return float(false_alarm), float(miss), float(total_error)
