"""AE 해석 — 피처별 재구성 오차 + SHAP.

⚠ AE 에서 "변수 중요도"는 회귀와 뜻이 다르다. 여기서는 두 가지를 낸다:
  1. **피처별 재구성 오차** — 어떤 신호를 AE 가 잘/못 복원하나.
     못 복원하는 피처는 잡음이 크거나 다른 피처로 설명이 안 되는 독립 정보다.
     → PINN 에 넘길 때 그 피처는 정제 효과를 기대하면 안 된다.
  2. **재구성오차에 대한 SHAP** — 어떤 입력이 '복원 난이도'를 좌우하나.
     스칼라 출력(총 재구성오차)으로 감싸서 GradientExplainer 를 태운다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
import torch

logger = logging.getLogger(__name__)


def per_feature_error(X_true: np.ndarray, X_rec: np.ndarray,
                      feature_names: list[str], scale: np.ndarray | None = None
                      ) -> pd.DataFrame:
    """피처별 RMSE/MAE. scale 을 주면 원 단위로 환산해 함께 보여준다."""
    err = X_rec - X_true
    rmse = np.sqrt((err ** 2).mean(axis=(0, 1)))
    mae = np.abs(err).mean(axis=(0, 1))
    df = pd.DataFrame({"feature": feature_names, "rmse": rmse, "mae": mae})
    # 신호 자체의 변동폭 대비 오차 — 스케일이 다른 피처를 공정하게 비교하려고
    sd = X_true.std(axis=(0, 1))
    df["nrmse_vs_sd"] = (rmse / np.where(sd > 1e-9, sd, 1.0)).round(4)
    if scale is not None:
        df["rmse_raw_unit"] = (rmse * scale).round(4)
    return df.sort_values("nrmse_vs_sd", ascending=False).reset_index(drop=True)


class _ReconError(torch.nn.Module):
    """AE 를 '입력 → 총 재구성오차(스칼라)' 로 감싼다. SHAP 은 2차원 출력을 기대한다."""

    def __init__(self, ae: torch.nn.Module) -> None:
        super().__init__()
        self.ae = ae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rec = self.ae(x)
        return ((rec - x) ** 2).mean(dim=(1, 2)).unsqueeze(-1)   # (N, 1)


def shap_on_error(ae, X_background: np.ndarray, X_explain: np.ndarray,
                  feature_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """재구성오차를 설명하는 SHAP. cuDNN 이중미분 미지원이라 CPU 에서 돈다."""
    wrapped = _ReconError(ae.to("cpu").eval()).eval()
    with torch.backends.cudnn.flags(enabled=False):
        sv = shap.GradientExplainer(wrapped, torch.from_numpy(X_background)) \
                 .shap_values(torch.from_numpy(X_explain))
    sv = sv[0] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    if sv.ndim == 4 and sv.shape[-1] == 1:
        sv = sv[..., 0]
    mean_abs = np.abs(sv).mean(axis=(0, 1))
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df["share_pct"] = 100 * df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    logger.info("SHAP(재구성오차) 완료: shap%s", sv.shape)
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True), sv
