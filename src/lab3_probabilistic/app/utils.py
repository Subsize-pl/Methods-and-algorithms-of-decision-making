import math
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
