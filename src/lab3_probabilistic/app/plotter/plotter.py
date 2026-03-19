from __future__ import annotations
import logging
from typing import Tuple
import matplotlib.pyplot as plt
from common.config.log import LoggingSettings
from lab3_probabilistic.app.plotter.plot_data import prepare_plot_data, PlotData

logging.basicConfig(level=LoggingSettings.LEVEL, format=LoggingSettings.FORMAT)
logger = logging.getLogger(__name__)


def _render_plot(
    data: PlotData,
    mu1: float,
    s1: float,
    mu2: float,
    s2: float,
    p1: float,
    p2: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(data.xs, data.p1_pdf, lw=2, label=f"p(x|C1) N({mu1},{s1:.2f})")
    ax.plot(data.xs, data.p2_pdf, lw=2, label=f"p(x|C2) N({mu2},{s2:.2f})")

    mask_fa = data.decisions == 1
    ax.fill_between(
        data.xs,
        0,
        data.p2_pdf,
        where=mask_fa,
        color="orange",
        alpha=0.35,
        label="False alarm",
    )

    mask_miss = data.decisions == 2
    ax.fill_between(
        data.xs,
        0,
        data.p1_pdf,
        where=mask_miss,
        color="red",
        alpha=0.25,
        label="Miss",
    )

    for r in data.roots:
        ax.axvline(r, color="k", linestyle="--", lw=1.2)

    if data.roots:
        ax.set_title(
            "Decision boundaries: " + ", ".join(f"{r:.3f}" for r in data.roots)
        )
    else:
        ax.set_title("No real decision boundary roots")

    ax.set_xlabel("x")
    ax.set_ylabel("pdf")
    ax.legend(loc="upper right")

    txt = (
        f"P_FA = {data.false_alarm:.6f}\n"
        f"P_miss = {data.miss:.6f}\n"
        f"Total error = {data.total_error:.6f}\n"
        f"Priors: P1={p1:.2f}, P2={p2:.2f}"
    )
    ax.text(
        0.02,
        0.95,
        txt,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", fc="w", alpha=0.8),
    )

    return fig


def make_plot(
    mu1: float,
    s1: float,
    mu2: float,
    s2: float,
    p1: float = 0.5,
    p2: float = 0.5,
    show: bool = True,
) -> Tuple[float, float, float]:
    """
    Build the plot data, render the graph, and optionally show it.
    Returns (false_alarm, miss, total_error).
    """
    data = prepare_plot_data(mu1, s1, mu2, s2, p1, p2)
    fig = _render_plot(data, mu1, s1, mu2, s2, p1, p2)

    try:
        if show:
            plt.show()
        else:
            plt.close(fig)
    except Exception:
        logger.exception("Plot display exception")

    return data.false_alarm, data.miss, data.total_error
