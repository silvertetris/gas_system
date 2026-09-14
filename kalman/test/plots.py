"""KF 결과 그림."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trend import style as trend_style  # noqa: E402

from . import config  # noqa: E402

logger = logging.getLogger(__name__)
FIGURE_DIR = config.OUTPUT_DIR / "figures"


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGURE_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("그림 저장: %s", p)
    return p


def plot_vs_baselines(met: pd.DataFrame) -> Path:
    trend_style.apply_style()
    d = met.sort_values("rmse")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["#2b6cb0" if m.startswith("KF") else "#a0aec0" for m in d["model"]]
    ax.barh(d["model"], d["rmse"], color=colors)
    for y, v in enumerate(d["rmse"]):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("1스텝 앞 예측 RMSE (원 단위 평균)")
    ax.set_title("KF vs 단순 평활기 — 못 이기면 상태공간 모델을 쓸 이유가 없다", fontsize=12)
    fig.tight_layout()
    return _save(fig, "01_kf_vs_baselines.png")


def plot_filtering(t, Z, res, signals, hours: int = 12) -> Path:
    """원신호 vs 필터링 결과 — 정제가 실제로 일어나는지 눈으로."""
    trend_style.apply_style()
    n = min(hours * 60, len(t))
    fig, axes = plt.subplots(len(signals), 1, figsize=(14, 1.9 * len(signals)), sharex=True)
    for i, s in enumerate(signals):
        axes[i].plot(t[:n], Z[:n, i], lw=0.8, color="#a0aec0", label="원신호")
        axes[i].plot(t[:n], res["filtered"][:n, i], lw=1.0, color="#2b6cb0", label="KF 필터")
        axes[i].set_ylabel(s, fontsize=8)
        if i == 0:
            axes[i].legend(fontsize=8, ncol=2)
    fig.suptitle(f"원신호 vs KF 필터링 (앞 {hours}시간) — 정제 효과 확인", fontsize=13)
    fig.tight_layout()
    return _save(fig, "02_filtering.png")


def plot_diagnostics(diag: pd.DataFrame) -> Path:
    """Kalman gain(계기 신뢰도) + innovation(모델 적합도) + NIS(보정 타당성)."""
    trend_style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    d = diag.sort_values("kalman_gain_mean")
    axes[0].barh(d["signal"], d["kalman_gain_mean"], color="#2b6cb0")
    axes[0].set_xlabel("평균 Kalman gain")
    axes[0].set_title("계기 신뢰도\n(1=계기를 믿음, 0=모델을 믿음)", fontsize=11)

    d2 = diag.sort_values("ninnov_vs_sd")
    axes[1].barh(d2["signal"], d2["ninnov_vs_sd"], color="#dd6b20")
    axes[1].set_xlabel("innovation RMSE / 신호 표준편차")
    axes[1].set_title("상태모델 적합도\n(작을수록 잘 따라감)", fontsize=11)

    d3 = diag.sort_values("nis_mean")
    axes[2].barh(d3["signal"], d3["nis_mean"], color="#718096")
    axes[2].axvline(1.0, color="#c53030", ls="--", lw=1.2)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("평균 NIS (이상적으로 1)")
    axes[2].set_title("Q/R 보정 타당성\n(1에서 멀면 잡음설정 오류)", fontsize=11)
    fig.suptitle("KF 신호별 진단 — SHAP 대신 KF 고유량으로", fontsize=13)
    fig.tight_layout()
    return _save(fig, "03_diagnostics.png")


def plot_innovation_timeline(t, res, signals) -> Path:
    """NIS 시계열 — 튀는 구간이 PINN 학습에서 뺄 후보."""
    trend_style.apply_style()
    nis = res["nis"]
    mean_nis = np.nanmean(nis, axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, mean_nis, lw=0.5, color="#2b6cb0")
    axes[0].axhline(3.84, color="#c53030", ls="--", lw=1,
                    label=f"χ²(1) 95% = 3.84 (초과 {100*np.nanmean(mean_nis>3.84):.1f}%)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("평균 NIS")
    axes[0].legend(fontsize=9)
    axes[0].set_title("정규화 innovation 제곱(NIS) — 상태모델이 깨지는 시점", fontsize=12)
    for i, s in enumerate(signals):
        axes[1].plot(t, np.nancumsum(nis[:, i] > 3.84), lw=1, label=s)
    axes[1].set_ylabel("누적 플래그 수")
    axes[1].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return _save(fig, "04_innovation.png")
