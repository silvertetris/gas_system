"""결측/데이터 커버리지 히트맵 (월 x 태그, 실측 비율)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .. import config, style, util

logger = logging.getLogger(__name__)


def plot_missing_heatmap(trend: pd.DataFrame) -> Path:
    """월별 x 태그별 실측(non-NaN) 비율 히트맵. 1분 그리드라 결측 = 계측 공백/미가동 구간."""
    style.apply_style()
    coverage = trend[config.TREND_TAGS].notna().resample("MS").mean()
    coverage.index = coverage.index.strftime("%Y-%m")

    fig, ax = plt.subplots(figsize=(max(10, len(coverage) * 0.14), 5.5))
    sns.heatmap(
        coverage.T,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "월별 실측 비율"},
        ax=ax,
        xticklabels=max(1, len(coverage) // 40),
    )
    ax.set_title("HTR-31P Trend 태그별 월간 데이터 커버리지 (결측 히트맵)", fontsize=13)
    ax.set_xlabel("월")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    fig.tight_layout()
    return util.save_fig(fig, "07_missing_heatmap.png")
