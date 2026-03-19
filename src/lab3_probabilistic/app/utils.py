import math
from typing import List

import numpy as np


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normal 'probability density function' evaluated elementwise"""
    coef = 1.0 / (math.sqrt(2 * math.pi) * sigma)
    e = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coef * e


def gaussian_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normal 'cumulative distribution function' via error function"""
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z))


def decision_boundary_roots(
    mu1: float, sigma1: float, mu2: float, sigma2: float, p1: float, p2: float
) -> List[float]:
    """
    Solve equation p1 * N(x|mu1,s1) = p2 * N(x|mu2,s2)
    Returns sorted real roots (0, 1 or 2 roots).
    """
    s1sq = sigma1**2
    s2sq = sigma2**2

    A = 1.0 / (2.0 * s2sq) - 1.0 / (2.0 * s1sq)
    B = mu1 / s1sq - mu2 / s2sq
    C = -0.5 * (mu1**2 / s1sq - mu2**2 / s2sq) + math.log((p1 * sigma2) / (p2 * sigma1))

    # Handle near-zero A (equal variances -> linear)
    eps = 1e-12
    roots: List[float] = []
    if abs(A) < eps:
        # linear B*x + C = 0
        if abs(B) > eps:
            roots.append(-C / B)
        return sorted(roots)

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return []
    if abs(disc) < 1e-14:
        roots.append(-B / (2.0 * A))
    else:
        sqrt_d = math.sqrt(disc)
        roots.append((-B + sqrt_d) / (2.0 * A))
        roots.append((-B - sqrt_d) / (2.0 * A))
    return sorted(roots)
