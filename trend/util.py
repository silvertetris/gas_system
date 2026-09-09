"""plot 모듈 공용 헬퍼."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from . import config

logger = logging.getLogger(__name__)


def save_fig(fig: plt.Figure, filename: str) -> Path:
    """FIGURE_DIR 아래에 PNG로 저장하고 figure를 닫는다."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
    logger.info("그림 저장: %s", path)
    return path
