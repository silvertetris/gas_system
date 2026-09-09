"""알람/이벤트(DI·DO) 발생 현황: 태그별 건수, 분류별 비중, 월별 추이."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .. import config, style, util

logger = logging.getLogger(__name__)


def plot_alarm_counts_by_tag(fault_events: pd.DataFrame) -> Path:
    """태그별 발생(상승엣지) 건수 가로 막대그래프. 분류(FAULT/STATUS/CONTROL)별로 색을 구분."""
    style.apply_style()
    counts = (
        fault_events.groupby(["tag", "category"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("count")
    )
    colors = counts["category"].map(config.CATEGORY_PALETTE).fillna("#999999")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(counts))))
    ax.barh(counts["tag"], counts["count"], color=colors)
    ax.set_xscale("symlog")
    ax.set_xlabel("발생 건수 (상승엣지, symlog 스케일)")
    ax.set_title(
        "HTR-31P 태그별 이벤트 발생 건수\n"
        "(H31POH 등 운전상태 STATUS 태그는 가동/정지마다 토글되어 FAULT보다 수천 배 많음)",
        fontsize=12,
    )

    for y, v in enumerate(counts["count"]):
        ax.text(v, y, f" {int(v):,}", va="center", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color) for color in config.CATEGORY_PALETTE.values()
    ]
    ax.legend(handles, config.CATEGORY_PALETTE.keys(), title="category", loc="lower right")
    fig.tight_layout()
    return util.save_fig(fig, "08_alarm_counts_by_tag.png")


def plot_alarm_category_share(fault_events: pd.DataFrame) -> Path:
    """분류(FAULT/STATUS/CONTROL)별 이벤트 발생 건수."""
    style.apply_style()
    counts = fault_events["category"].value_counts().reindex(config.CATEGORY_ORDER).fillna(0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [config.CATEGORY_PALETTE.get(c, "#999999") for c in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("발생 건수")
    ax.set_title("분류별 이벤트 발생 건수", fontsize=13)
    fig.tight_layout()
    return util.save_fig(fig, "09_alarm_category_share.png")


def plot_alarm_monthly_trend(fault_events: pd.DataFrame) -> Path:
    """FAULT 분류 이벤트의 월별 발생 건수 추이 (설비 열화/이상징후 추이 파악용)."""
    style.apply_style()
    fault_only = fault_events[fault_events["category"] == "FAULT"].set_index("Time")
    monthly = fault_only.resample("MS").size()

    fig, ax = plt.subplots(figsize=(max(10, len(monthly) * 0.09), 4.5))
    ax.bar(monthly.index, monthly.values, width=20, color=config.CATEGORY_PALETTE["FAULT"])
    ax.set_ylabel("FAULT 이벤트 건수")
    ax.set_title("HTR-31P FAULT 이벤트 월별 발생 추이", fontsize=13)
    fig.tight_layout()
    return util.save_fig(fig, "10_alarm_monthly_trend.png")
