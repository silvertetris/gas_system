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


def plot_fault_catalog(fault_events: pd.DataFrame) -> Path:
    """FAULT 태그 카탈로그 — 빈도 + **설명 + 계통 + 예측가치**를 한 장에.

    기존 `08_alarm_counts_by_tag.png` 은 태그 코드만 찍어서 `TALD1P` 가 수조인지
    버너인지 알 수 없었다. 이 그림은 세 가지를 같이 보여준다:
      - 계통별 색 (수조/연소/연료가스/차단밸브/안전)
      - 막대 옆에 한글 설명
      - 문서 04 §2 의 예측가치 별점
    """
    style.apply_style()
    f = fault_events[fault_events["category"] == "FAULT"]
    g = (f.groupby("tag")
           .agg(건수=("Time", "size"), 시작=("Time", "min"), 종료=("Time", "max"))
           .reset_index().sort_values("건수"))
    g["계통"] = g["tag"].map(config.FAULT_SUBSYSTEM).fillna("기타")
    g["설명"] = g["tag"].map(lambda t: config.ALARM_TAG_META.get(t, {}).get("description", ""))
    g["가치"] = g["tag"].map(config.FAULT_PRIORITY).fillna("")

    palette = {"수조": "#2b6cb0", "연소": "#e53e3e", "연료가스": "#dd6b20",
               "차단밸브": "#805ad5", "안전(누출)": "#38a169", "기타": "#a0aec0"}
    fig, ax = plt.subplots(figsize=(14, 0.55 * len(g) + 2.5))
    ax.barh(g["tag"], g["건수"], color=[palette[s] for s in g["계통"]])
    xmax = g["건수"].max()
    for y, r in enumerate(g.itertuples()):
        ax.text(r.건수 * 1.05, y, f"{r.건수:,}  [{r.계통}] {r.설명}  {r.가치}",
                va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1, xmax * 25)
    ax.set_xlabel("발생 건수 (상승엣지, log 스케일)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), title="계통", loc="lower right", fontsize=9)
    total = int(g["건수"].sum())
    ax.set_title(f"HTR-31P FAULT 태그 카탈로그 — 총 {total:,}건 (2011~2026)\n"
                 f"별점 = 문서 04 §2 예측가치", fontsize=13)
    fig.tight_layout()
    return util.save_fig(fig, "11_fault_catalog.png")


def plot_subsystem_share(fault_events: pd.DataFrame) -> Path:
    """계통별 비중 + 연도별 추이 — 어느 계통이 문제인가."""
    style.apply_style()
    f = fault_events[fault_events["category"] == "FAULT"].copy()
    f["계통"] = f["tag"].map(config.FAULT_SUBSYSTEM).fillna("기타")
    palette = {"수조": "#2b6cb0", "연소": "#e53e3e", "연료가스": "#dd6b20",
               "차단밸브": "#805ad5", "안전(누출)": "#38a169", "기타": "#a0aec0"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    cnt = f["계통"].value_counts()
    axes[0].pie(cnt, labels=[f"{k}\n{v:,}건 ({100*v/cnt.sum():.1f}%)" for k, v in cnt.items()],
                colors=[palette[k] for k in cnt.index], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title("계통별 FAULT 비중", fontsize=12)

    yr = f.groupby([f["Time"].dt.year, "계통"]).size().unstack(fill_value=0)
    yr.plot(kind="bar", stacked=True, ax=axes[1],
            color=[palette[c] for c in yr.columns], width=0.85)
    axes[1].set_xlabel("연도")
    axes[1].set_ylabel("FAULT 건수")
    axes[1].legend(fontsize=8, title="계통")
    axes[1].set_title("연도별 계통 구성", fontsize=12)
    fig.tight_layout()
    return util.save_fig(fig, "12_fault_subsystem.png")
