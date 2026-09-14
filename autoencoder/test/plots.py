"""AE 결과 그림."""
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


def plot_vs_pca(ae_rmse: float, pca: pd.DataFrame, latent_dim: int) -> Path:
    """AE vs PCA — 같은 압축 차원에서 비선형이 이득이 있나."""
    trend_style.apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(pca["n_components"], pca["rmse"], "o-", color="#a0aec0", lw=2, ms=7, label="PCA")
    ax.scatter([latent_dim], [ae_rmse], s=140, color="#2b6cb0", zorder=5,
               label=f"LSTM-AE (latent={latent_dim})")
    same = pca.loc[pca["n_components"] == latent_dim, "rmse"]
    if len(same):
        gain = 100 * (same.iloc[0] - ae_rmse) / same.iloc[0]
        ax.annotate(f"{gain:+.1f}%", (latent_dim, ae_rmse), textcoords="offset points",
                    xytext=(10, -14), color="#2b6cb0", fontsize=11)
    ax.set_xlabel("압축 차원 (latent_dim / n_components)")
    ax.set_ylabel("테스트 재구성 RMSE (원 단위)")
    ax.set_title("AE vs PCA — PCA 를 못 이기면 비선형을 쓸 이유가 없다", fontsize=12)
    ax.legend()
    fig.tight_layout()
    return _save(fig, "01_ae_vs_pca.png")


def plot_feature_error(err: pd.DataFrame) -> Path:
    """피처별 재구성 난이도. 값이 크면 AE 가 그 신호를 설명하지 못한다는 뜻."""
    trend_style.apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    d = err.sort_values("nrmse_vs_sd")
    colors = ["#c53030" if v > 0.5 else "#2b6cb0" for v in d["nrmse_vs_sd"]]
    ax.barh(d["feature"], d["nrmse_vs_sd"], color=colors)
    for y, v in enumerate(d["nrmse_vs_sd"]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, color="#c53030", ls="--", lw=1)
    ax.set_xlabel("재구성 RMSE / 신호 표준편차  (0=완벽, 1=평균만 찍는 수준)")
    ax.set_title("피처별 재구성 난이도 — 붉은색은 AE 가 설명 못 하는 신호", fontsize=12)
    fig.tight_layout()
    return _save(fig, "02_feature_error.png")


def plot_reconstruction(X_true: np.ndarray, X_rec: np.ndarray, features: list[str],
                        n_show: int = 3) -> Path:
    """임의 윈도우 몇 개를 골라 원본 vs 재구성 파형 비교."""
    trend_style.apply_style()
    rng = np.random.default_rng(config.SEED)
    picks = rng.choice(len(X_true), min(n_show, len(X_true)), replace=False)
    nf = len(features)
    fig, axes = plt.subplots(nf, len(picks), figsize=(4.4 * len(picks), 1.7 * nf),
                             sharex=True)
    for c, i in enumerate(picks):
        for r, f in enumerate(features):
            ax = axes[r, c] if len(picks) > 1 else axes[r]
            ax.plot(X_true[i, :, r], lw=1.0, color="#2d3748", label="원본")
            ax.plot(X_rec[i, :, r], lw=1.0, color="#e53e3e", alpha=0.85, label="재구성")
            if c == 0:
                ax.set_ylabel(f, fontsize=8)
            if r == 0:
                ax.set_title(f"윈도우 #{i}", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle("원본 vs AE 재구성 (스케일 단위)", fontsize=13)
    fig.tight_layout()
    return _save(fig, "03_reconstruction.png")


def plot_error_timeline(t: pd.DatetimeIndex, err: np.ndarray) -> Path:
    """재구성오차 시계열 — 튀는 구간이 PINN 학습에서 뺄 후보다."""
    trend_style.apply_style()
    s = pd.Series(err, index=t)
    thr = s.median() + 3 * (s.quantile(.75) - s.quantile(.25))    # robust 임계
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(s.index, s.values, lw=0.5, color="#2b6cb0")
    axes[0].axhline(thr, color="#c53030", ls="--", lw=1,
                    label=f"robust 임계 {thr:.4f} (초과 {100*(s>thr).mean():.1f}%)")
    axes[0].set_ylabel("윈도우 재구성오차 (MSE)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].set_title("재구성오차 시계열 — 튀는 구간 = PINN 학습에서 제외 후보", fontsize=12)
    axes[1].hist(np.log10(s.values + 1e-12), bins=80, color="#2b6cb0")
    axes[1].axvline(np.log10(thr), color="#c53030", ls="--")
    axes[1].set_xlabel("log10(재구성오차)")
    fig.tight_layout()
    return _save(fig, "04_error_timeline.png")


def plot_latent(z: np.ndarray, t: pd.DatetimeIndex) -> Path:
    """잠재벡터 궤적 — PINN 상태표현 후보가 해석 가능한 모양인지."""
    trend_style.apply_style()
    k = min(z.shape[1], 6)
    fig, axes = plt.subplots(k, 1, figsize=(14, 1.5 * k), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(k):
        axes[i].plot(t, z[:, i], lw=0.5, color="#2b6cb0")
        axes[i].set_ylabel(f"z{i}", fontsize=9)
    fig.suptitle(f"잠재벡터 궤적 (latent_dim={z.shape[1]}, 앞 {k}개) — PINN 상태표현 후보", fontsize=13)
    fig.tight_layout()
    return _save(fig, "05_latent.png")


def plot_shap(imp: pd.DataFrame) -> Path:
    trend_style.apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    d = imp.sort_values("mean_abs_shap")
    ax.barh(d["feature"], d["mean_abs_shap"], color="#2b6cb0")
    for y, (v, s) in enumerate(zip(d["mean_abs_shap"], d["share_pct"])):
        ax.text(v, y, f" {s:.1f}%", va="center", fontsize=9)
    ax.set_title("재구성오차에 대한 SHAP — 어떤 입력이 복원 난이도를 좌우하나", fontsize=12)
    fig.tight_layout()
    return _save(fig, "06_shap.png")
