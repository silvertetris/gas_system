"""LSTM 파일럿 결과 그림.

trend/ 와 같은 규칙: 한글 폰트, FIGURE_DIR 저장, figure 닫기.
"""
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


def plot_metrics(met: pd.DataFrame) -> Path:
    """LSTM vs 베이스라인 막대. persistence 를 못 이기면 모델이 의미 없다."""
    trend_style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col in zip(axes, ["RMSE", "MAE"]):
        d = met.sort_values(col)
        colors = ["#2b6cb0" if m == "lstm" else "#a0aec0" for m in d["model"]]
        ax.bar(d["model"], d[col], color=colors)
        for x, v in zip(d["model"], d[col]):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{col} (℃, 낮을수록 좋음)")
    fig.suptitle("LSTM vs 무학습 베이스라인 — TI33P 60분 예측", fontsize=13)
    fig.tight_layout()
    return _save(fig, "01_metrics.png")


def plot_prediction(pred: pd.DataFrame, hours: int = 72) -> Path:
    """테스트 구간 실측 vs 예측 (앞부분 확대 + 전체)."""
    trend_style.apply_style()
    d = pred.copy()
    d["target_time"] = pd.to_datetime(d["target_time"])
    head = d.head(hours * 60)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, g, title in [(axes[0], head, f"테스트 앞 {hours}시간 (확대)"),
                         (axes[1], d, "테스트 전 구간")]:
        ax.plot(g["target_time"], g["actual"], lw=0.9, color="#2d3748", label="실측 TI33P")
        ax.plot(g["target_time"], g["lstm"], lw=0.9, color="#e53e3e", alpha=0.85, label="LSTM 예측")
        ax.plot(g["target_time"], g["persistence"], lw=0.6, color="#a0aec0",
                alpha=0.7, label="persistence")
        ax.set_ylabel("℃")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("HTR-31P 출구온도 60분 예측 — 실측 vs 예측", fontsize=13)
    fig.tight_layout()
    return _save(fig, "02_prediction.png")


def plot_residual(pred: pd.DataFrame) -> Path:
    """잔차 분포·시간추이·실측대비 산점도. 이상탐지의 재료가 잔차다."""
    trend_style.apply_style()
    d = pred.copy()
    d["target_time"] = pd.to_datetime(d["target_time"])
    r = d["residual_lstm"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].hist(r, bins=80, color="#2b6cb0", alpha=0.85)
    axes[0].axvline(0, color="#c53030", ls="--", lw=1)
    axes[0].set_title(f"잔차 분포 (평균 {r.mean():+.3f}, σ {r.std():.2f}℃)")
    axes[0].set_xlabel("실측 − 예측 [℃]")

    axes[1].plot(d["target_time"], r, lw=0.4, color="#2b6cb0")
    axes[1].axhline(0, color="#c53030", ls="--", lw=1)
    for k, c in [(2, "#dd6b20"), (3, "#c53030")]:
        axes[1].axhline(k * r.std(), color=c, ls=":", lw=0.8)
        axes[1].axhline(-k * r.std(), color=c, ls=":", lw=0.8)
    axes[1].set_title("잔차 시간추이 (±2σ/±3σ)")

    axes[2].scatter(d["actual"], d["lstm"], s=2, alpha=0.15, color="#2b6cb0")
    lo, hi = d["actual"].min(), d["actual"].max()
    axes[2].plot([lo, hi], [lo, hi], color="#c53030", ls="--", lw=1)
    axes[2].set_xlabel("실측 [℃]")
    axes[2].set_ylabel("예측 [℃]")
    axes[2].set_title("실측 vs 예측")
    fig.suptitle("LSTM 잔차 진단", fontsize=13)
    fig.tight_layout()
    return _save(fig, "03_residual.png")


def plot_shap(imp: pd.DataFrame, prof: pd.DataFrame) -> Path:
    """피처별 중요도 + 윈도우 내 시간축 프로파일."""
    trend_style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    d = imp.sort_values("mean_abs_shap")
    axes[0].barh(d["feature"], d["mean_abs_shap"], color="#2b6cb0")
    for y, (v, s) in enumerate(zip(d["mean_abs_shap"], d["share_pct"])):
        axes[0].text(v, y, f" {s:.1f}%", va="center", fontsize=9)
    axes[0].set_title("피처별 SHAP 중요도 (|shap| 평균)")

    for c in prof.columns:
        axes[1].plot(prof.index, prof[c], lw=1.2, label=c)
    axes[1].set_xlabel("윈도우 내 시점 (0 = 예측 시작 직전)")
    axes[1].set_ylabel("|shap| 평균")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8, ncol=2)
    share = prof.sum(axis=1)
    recent = 100 * share.loc[-29:].sum() / share.sum()
    axes[1].set_title(f"시간축 프로파일 — 최근 30분이 기여의 {recent:.1f}%")
    fig.suptitle("SHAP 변수 중요도", fontsize=13)
    fig.tight_layout()
    return _save(fig, "04_shap.png")


def plot_optuna(trials: pd.DataFrame) -> Path:
    """탐색 이력 + 주요 하이퍼파라미터별 성능."""
    trend_style.apply_style()
    d = trials[trials["state"] == "COMPLETE"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    axes[0].plot(d["number"], d["value"], "o-", ms=4, color="#a0aec0", lw=0.8)
    axes[0].plot(d["number"], d["value"].cummin(), color="#2b6cb0", lw=2, label="최고 기록")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("CV RMSE [℃]")
    axes[0].set_title("탐색 이력")
    axes[0].legend(fontsize=9)

    axes[1].scatter(d["params_lr"], d["value"], s=28, c=d["params_hidden_size"], cmap="viridis")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel("CV RMSE [℃]")
    axes[1].set_title("lr vs 성능 (색=hidden_size)")

    grp = d.groupby("params_hidden_size")["value"]
    axes[2].bar(grp.mean().index.astype(str), grp.mean(), yerr=grp.std(),
                color="#2b6cb0", capsize=4)
    axes[2].set_xlabel("hidden_size")
    axes[2].set_ylabel("CV RMSE [℃]")
    axes[2].set_title("hidden_size별 평균 성능")
    fig.suptitle(f"Optuna 탐색 ({len(d)} trials × {config.N_SPLITS}-fold TimeSeriesSplit)", fontsize=13)
    fig.tight_layout()
    return _save(fig, "05_optuna.png")


def plot_window_comparison(res: pd.DataFrame) -> Path:
    """윈도우 길이별 성능 — 긴 윈도우가 실제로 도움이 되나."""
    trend_style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(res["window"], res["RMSE"], "o-", color="#2b6cb0", lw=2, ms=7)
    for x, v in zip(res["window"], res["RMSE"]):
        axes[0].text(x, v, f" {v:.2f}", fontsize=9)
    axes[0].axhline(res["persistence_RMSE"].iloc[0], color="#a0aec0", ls="--",
                    label=f"persistence {res['persistence_RMSE'].iloc[0]:.2f}")
    axes[0].set_xlabel("입력 윈도우 [분]")
    axes[0].set_ylabel("테스트 RMSE [℃]")
    axes[0].set_title("윈도우 길이 vs 예측 성능")
    axes[0].legend(fontsize=9)

    axes[1].bar(res["window"].astype(str), res["train_sec"], color="#718096")
    axes[1].set_xlabel("입력 윈도우 [분]")
    axes[1].set_ylabel("최종 학습 시간 [초]")
    axes[1].set_title("학습 비용")
    tau = 63
    axes[0].axvline(tau, color="#dd6b20", ls=":", lw=1.5)
    axes[0].text(tau, axes[0].get_ylim()[1], " τ=63분", color="#dd6b20", fontsize=9, va="top")
    fig.suptitle("입력 윈도우 길이 비교 (하이퍼파라미터 고정)", fontsize=13)
    fig.tight_layout()
    return _save(fig, "06_window_comparison.png")
