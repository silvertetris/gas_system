"""전체 기간 시계열 개요 (태그별 일평균 line plot)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .. import config, style, util

logger = logging.getLogger(__name__)


def plot_time_series_overview(trend: pd.DataFrame) -> Path:
    """태그별 일평균을 태그당 1개 subplot으로 쌓아 전체 기간(2011~) 추이를 한눈에 본다.

    원시 1분 그리드(수백만 행)를 그대로 그리면 느리고 노이즈에 묻히므로 일단위로 리샘플한다.
    """
    style.apply_style()
    daily = trend[config.TREND_TAGS].resample("1D").mean()

    n = len(config.TREND_TAGS)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    for ax, tag in zip(axes, config.TREND_TAGS):
        ax.plot(daily.index, daily[tag], linewidth=0.6, color="#2b6cb0")
        ax.set_ylabel(config.TAG_LABELS.get(tag, tag), fontsize=9)
        ax.margins(x=0)
    axes[-1].set_xlabel("Time")
    fig.suptitle("HTR-31P Trend 태그별 일평균 추이 (전체 기간)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return util.save_fig(fig, "01_time_series_overview.png")
