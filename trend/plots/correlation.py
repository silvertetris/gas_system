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
    """상관계수 히트맵 — 원본 태그 + 파생컬럼 16개(config.ANALYSIS_TAGS).

    Pearson과 Spearman을 나란히 그린다. 이 데이터는 왜도가 커서(RSF41P skew −8.8,
    PI21X −4.5 등, prex/profiling.py) 선형상관만 보면 오독한다.
    결측은 **쌍별(pairwise)** 로 제외한다 — htx_eps 가 64%만 유효해서 행 dropna 를 하면
    표본이 통째로 날아간다. 쌍별 유효표본수는 prex/output/profile/corr_pearson_n.csv 참고.
    """
    style.apply_style()
    cols = [c for c in config.ANALYSIS_TAGS if c in trend.columns]
    sub = trend[cols]
    mats = [("Pearson", sub.corr(method="pearson")),
            ("Spearman", sub.sample(min(config.CORR_SPEARMAN_SAMPLE, len(sub)),
                                    random_state=config.RANDOM_SEED).corr(method="spearman"))]

    fig, axes = plt.subplots(1, 2, figsize=(23, 10))
    for ax, (name, corr) in zip(axes, mats):
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", annot_kws={"fontsize": 7},
            cmap="RdBu_r", vmin=-1, vmax=1, square=True,
            cbar_kws={"label": f"{name} r", "shrink": 0.7}, ax=ax,
            xticklabels=cols, yticklabels=cols,
        )
        ax.set_title(f"{name} 상관계수", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.suptitle(f"HTR-31P Trend 상관계수 히트맵 — 원본+파생 {len(cols)}개 (결측 쌍별 제외)", fontsize=14)
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
