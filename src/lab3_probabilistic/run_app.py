from __future__ import annotations

import logging

from common.config.log import LoggingSettings
from app.plotter import make_plot

logging.basicConfig(level=LoggingSettings.LEVEL, format=LoggingSettings.FORMAT)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    mu1, s1 = 0.0, 1.0
    mu2, s2 = 4.0, 1.0
    p1, p2 = 0.5, 0.5

    logger.info("Running: probabilistic 1D classification")
    false_alarm, miss, total = make_plot(mu1, s1, mu2, s2, p1=p1, p2=p2, show=True)
    logger.info(
        "Results: P_FA=%.6f, P_miss=%.6f, Total=%.6f",
        false_alarm,
        miss,
        total,
    )
