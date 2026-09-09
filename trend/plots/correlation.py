"""태그 간 상관관계: 상관계수 히트맵, 산점도 행렬(pairplot)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import config, style, util

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(trend: pd.DataFrame) -> Path:
    """태그 간 피어슨 상관계수 히트맵. 전체 데이터로 계산(결측은 쌍별 제외)."""
    style.apply_style()
    corr = trend[config.TREND_TAGS].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
        xticklabels=[t for t in config.TREND_TAGS],
        yticklabels=[t for t in config.TREND_TAGS],
    )
    ax.set_title("HTR-31P Trend 태그 상관계수 히트맵", fontsize=13)
    fig.tight_layout()
    return util.save_fig(fig, "05_correlation_heatmap.png")


def plot_pairplot(trend: pd.DataFrame) -> Path:
    """태그 쌍별 산점도 + 대각선 분포(산점도 행렬). 무거운 그림이라 샘플링한다."""
    style.apply_style()
    df = trend[config.TREND_TAGS].dropna()
    if len(df) > config.PAIRPLOT_SAMPLE_MAX:
        df = df.sample(config.PAIRPLOT_SAMPLE_MAX, random_state=config.RANDOM_SEED)

    grid = sns.pairplot(
        df,
        diag_kind="kde",
        plot_kws={"s": 6, "alpha": 0.25, "color": "#2b6cb0"},
        diag_kws={"color": "#2b6cb0"},
    )
    grid.figure.suptitle(
        f"HTR-31P Trend 태그 산점도 행렬 (n={len(df):,} 샘플링)", fontsize=13, y=1.02
    )
    return util.save_fig(grid.figure, "06_pairplot.png")
