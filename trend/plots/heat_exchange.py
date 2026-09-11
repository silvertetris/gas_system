"""열교환 지표(ε · NTU) 전용 그림 — 모델 관측량을 눈으로 확인한다.

ε = (T_out − T_in)/(T_bath − T_in),  NTU = −ln(1−ε) = U·A/(m_gas·c_p)
계산 위치: prex/trend.py::add_heat_exchange().  설계 맥락: docs/kfpinn_state_space.md §2.

이 값들이 왜 따로 그림을 갖나:
  - ε 는 KF-PINN 의 **관측 z₃** 이고 NTU 는 물리식 H3 의 좌변이다. 즉 모델이 실제로 먹는 값인데
    기존 EDA(01~10)는 원본 6태그만 봐서 한 장도 없었다.
  - ε 는 `PI43O`·`ZI41P`·`Cv` 를 안 쓰는 유일한 경로라, P&ID 로 흔들린 재료(htr31p_flow.md §7)의
    영향을 받지 않는다. 그래서 여기서 이상이 보이면 그건 계기/설비 문제지 가정 문제가 아니다.

⚠ ε 는 NTU 의 단조변환이라 둘의 Spearman 상관은 정확히 1.000 이다(prex/profiling.py 확인).
   같은 정보를 두 축으로 보는 것뿐이니 모델 입력에 둘 다 넣지 말 것.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import config, style, util

logger = logging.getLogger(__name__)

BURNER_TAG = "H31POH"
BURNER_ON_VALUE = 1.0


def burner_state(alarm_events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """H31POH SET/RESET 이벤트를 1분 그리드에 전파해 ON/OFF/unknown 라벨을 만든다.

    prex/verify_ua.py::burner_state_on_grid 와 같은 방식이다(직전 상태 ffill).
    """
    h = alarm_events.loc[alarm_events["tag"] == BURNER_TAG, ["Time", "value"]].dropna()
    if h.empty:
        return pd.Series("unknown", index=index, name="burner")
    h = h.sort_values("Time").drop_duplicates("Time", keep="last")
    merged = pd.merge_asof(pd.DataFrame({"Time": index}), h, on="Time", direction="backward")
    v = pd.Series(merged["value"].to_numpy(), index=index)
    return v.map(lambda x: "unknown" if pd.isna(x) else ("ON" if x == BURNER_ON_VALUE else "OFF")).rename("burner")


def plot_eps_timeseries(trend: pd.DataFrame) -> Path:
    """ε·NTU 일별 중앙값 추이 + 유효표본 비율. 장기 하락이 있으면 열화 후보다."""
    style.apply_style()
    daily = trend[["htx_eps", "htx_ntu"]].resample("1D").median()
    valid = trend["htx_eps"].notna().resample("1D").mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(daily.index, daily["htx_eps"], lw=0.6, color="#2b6cb0")
    axes[0].set_ylabel("ε (일 중앙값)")
    axes[0].set_title("열교환 효율 ε · NTU 일별 추이 — HTR-31P", fontsize=13)

    axes[1].plot(daily.index, daily["htx_ntu"], lw=0.6, color="#b7791f")
    axes[1].set_ylabel("NTU (일 중앙값)")

    axes[2].fill_between(valid.index, valid.to_numpy(), color="#718096", alpha=0.6)
    axes[2].set_ylabel("ε 유효 비율")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("날짜")
    axes[2].axhline(0.5, color="#c53030", ls="--", lw=0.8)
    fig.tight_layout()
    return util.save_fig(fig, "11_eps_ntu_timeseries.png")


def plot_eps_by_burner(trend: pd.DataFrame, alarm_events: pd.DataFrame) -> Path:
    """버너 ON/OFF 별 ε·NTU·구동온도차 분포. ON/OFF 로 값이 갈리면 부하 의존을 뜻한다."""
    style.apply_style()
    df = trend[["htx_eps", "htx_ntu", "htx_drive_c"]].copy()
    df["burner"] = burner_state(alarm_events, trend.index)
    df = df[df["burner"].isin(["ON", "OFF"]) & df["htx_eps"].notna()]
    if len(df) > config.DIST_SAMPLE_MAX:
        df = df.sample(config.DIST_SAMPLE_MAX, random_state=config.RANDOM_SEED)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    palette = {"ON": "#dd6b20", "OFF": "#4a5568"}
    for ax, col, title in zip(
        axes,
        ["htx_eps", "htx_ntu", "htx_drive_c"],
        ["열교환 효율 ε", "NTU", "구동 온도차 T_bath−T_in [℃]"],
    ):
        sns.boxplot(data=df, x="burner", y=col, hue="burner", order=["ON", "OFF"],
                    palette=palette, legend=False, ax=ax, fliersize=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(f"버너 ON/OFF 별 열교환 지표 분포 (n={len(df):,} 샘플링)", fontsize=13)
    fig.tight_layout()
    return util.save_fig(fig, "12_eps_by_burner.png")


def plot_eps_yearly(trend: pd.DataFrame, alarm_events: pd.DataFrame) -> Path:
    """연도별 ε 중앙값 — 열화 신호 후보.

    ⚠ 해석 주의: NTU = U·A/(m_gas·c_p) 이므로 ε 하락은 **열화(U·A↓)일 수도, 부하 증가(m_gas↑)일
    수도** 있다. 둘은 이 그림만으로 안 갈린다 — 그래서 KF-PINN 에서 부하보정
    (U·A = UA_ref·(m/m_design)^0.8) 후 UA_ref 를 봐야 한다. docs/kfpinn_state_space.md §3·§4.
    """
    style.apply_style()
    df = trend[["htx_eps"]].copy()
    df["burner"] = burner_state(alarm_events, trend.index)
    df["year"] = df.index.year
    df = df[df["htx_eps"].notna()]

    stat = (df.groupby(["year", "burner"])["htx_eps"]
              .agg(["median", "count"]).reset_index())
    stat = stat[stat["burner"].isin(["ON", "OFF"]) & (stat["count"] >= 1000)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for burner, g in stat.groupby("burner"):
        axes[0].plot(g["year"], g["median"], marker="o",
                     color={"ON": "#dd6b20", "OFF": "#4a5568"}[burner], label=f"버너 {burner}")
    axes[0].set_ylabel("ε 중앙값")
    axes[0].legend()
    axes[0].set_title("연도별 열교환 효율 ε 중앙값 — 하락 시 열화 후보 (단, 부하 영향과 미분리)", fontsize=13)

    # 막대도 **수치 x축**으로 직접 그린다. pandas .plot(kind="bar") 는 x를 범주형(0,1,2…)으로
    # 바꿔 버려서 sharex=True 로 묶인 위 선그래프(x=연도)가 축 밖으로 밀려 사라진다.
    pivot = stat.pivot(index="year", columns="burner", values="count").fillna(0)
    bottom = np.zeros(len(pivot))
    for col in pivot.columns:
        axes[1].bar(pivot.index, pivot[col], bottom=bottom,
                    color={"ON": "#dd6b20", "OFF": "#4a5568"}[col], width=0.7)
        bottom += pivot[col].to_numpy()
    axes[1].set_ylabel("유효 표본수")
    axes[1].set_xlabel("연도")
    axes[1].set_xticks(pivot.index)
    axes[1].set_xticklabels(pivot.index, rotation=90)
    fig.tight_layout()
    return util.save_fig(fig, "13_eps_yearly.png")
