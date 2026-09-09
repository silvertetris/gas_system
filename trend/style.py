"""Matplotlib/Seaborn 공통 스타일 설정 (한글 폰트 포함)."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from . import config

logger = logging.getLogger(__name__)

_applied = False


def apply_style() -> None:
    """모든 plot 모듈에서 최초 1회 호출. seaborn 테마 + 한글 폰트 적용."""
    global _applied
    if _applied:
        return

    sns.set_theme(style="whitegrid", palette="deep")

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in config.FONT_FAMILY_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        logger.warning(
            "한글 폰트를 찾지 못했습니다 (후보: %s). 라벨이 깨져 보일 수 있습니다.",
            config.FONT_FAMILY_CANDIDATES,
        )

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = config.FIG_DPI
    plt.rcParams["savefig.bbox"] = "tight"
    _applied = True
