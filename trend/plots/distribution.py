"""태그별 분포: 히스토그램, 박스플롯."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from prex import config as prex_config

from .. import config, style, util

logger = logging.getLogger(__name__)


def _sample(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if len(s) > config.DIST_SAMPLE_MAX:
        s = s.sample(config.DIST_SAMPLE_MAX, random_state=config.RANDOM_SEED)
    return s


def _grid_shape(n: int) -> tuple[int, int]:
    ncols = 3
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_histograms(trend: pd.DataFrame) -> Path:
    """태그별 히스토그램(+KDE). 단위가 서로 달라 태그마다 독립된 축을 쓴다."""
    style.apply_style()
    tags = config.TREND_TAGS
    nrows, ncols = _grid_shape(len(tags))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = axes.flatten()
    for ax, tag in zip(axes, tags):
        sns.histplot(_sample(trend[tag]), kde=True, ax=ax, color="#2b6cb0")
        ax.set_title(config.TAG_LABELS.get(tag, tag).replace("\n", " "), fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(tags):]:
        ax.axis("off")
    fig.suptitle("HTR-31P Trend 태그별 분포 (히스토그램, 샘플링)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return util.save_fig(fig, "02_histograms.png")


def plot_boxplots(trend: pd.DataFrame) -> Path:
    """태그별 박스플롯(이상치 확인용). 단위가 달라 태그마다 독립된 축을 쓴다."""
    style.apply_style()
    tags = config.TREND_TAGS
    nrows, ncols = _grid_shape(len(tags))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4 * nrows))
    axes = axes.flatten()
    for ax, tag in zip(axes, tags):
        sns.boxplot(y=_sample(trend[tag]), ax=ax, color="#63b3ed", fliersize=2)
        ax.set_title(config.TAG_LABELS.get(tag, tag).replace("\n", " "), fontsize=10)
        ax.set_ylabel("")
    for ax in axes[len(tags):]:
        ax.axis("off")
    fig.suptitle("HTR-31P Trend 태그별 박스플롯 (샘플링)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return util.save_fig(fig, "03_boxplots.png")


def plot_pi_d2p_by_era(trend: pd.DataFrame) -> Path:
    """PI-D2P는 계기 스팬이 시기(era)별로 달라 원본/정규화값을 era별 박스플롯으로 비교한다.

    문서 근거: prex/config.py PI_D2P_ERAS (연 단위 근사 경계).
    """
    style.apply_style()
    df = trend[["PI-D2P", "PI-D2P_norm", "PI-D2P_era"]].dropna(subset=["PI-D2P_era"])
    sampled = df if len(df) <= config.DIST_SAMPLE_MAX else df.sample(
        config.DIST_SAMPLE_MAX, random_state=config.RANDOM_SEED
    )
    era_order = [label for label, _, _ in prex_config.PI_D2P_ERAS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=sampled, x="PI-D2P_era", y="PI-D2P", order=era_order, ax=axes[0], color="#f6ad55")
    axes[0].set_title("PI-D2P 원본값 (계기 스팬 era별)")
    axes[0].set_xlabel("era")

    sns.boxplot(data=sampled, x="PI-D2P_era", y="PI-D2P_norm", order=era_order, ax=axes[1], color="#68d391")
    axes[1].set_title("PI-D2P_norm (era-wise z-score 정규화 후)")
    axes[1].set_xlabel("era")

    fig.suptitle("PI-D2P 계기 스팬 변화 확인 (era별 박스플롯)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return util.save_fig(fig, "04_pi_d2p_era_boxplot.png")
