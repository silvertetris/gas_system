"""한글 폰트·스타일."""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from . import config

logger = logging.getLogger(__name__)
_applied = False


def apply() -> str:
    global _applied
    avail = {f.name for f in font_manager.fontManager.ttflist}
    pick = next((f for f in config.FONT_CANDIDATES if f in avail), "DejaVu Sans")
    if not _applied:
        logger.info("폰트: %s (후보 중 사용가능: %s)", pick,
                    [f for f in config.FONT_CANDIDATES if f in avail])
        _applied = True
    plt.rcParams.update({
        "font.family": pick,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    return pick
