"""SHAP 변수 중요도.

시퀀스 입력 (N, L, F) 이라 SHAP 값도 (N, L, F) 로 나온다.
**시간축으로 |shap| 을 합쳐** 피처별 중요도를 만든다 — "이 변수가 윈도우 전체에서
예측에 얼마나 기여했나"가 우리가 알고 싶은 것이기 때문이다.

⚠ GPU 위의 cuDNN LSTM 은 SHAP 이 필요로 하는 이중 미분을 지원하지 않는다.
   그래서 설명 단계는 **CPU 로 옮겨서** 수행한다(파일럿 규모라 비용이 크지 않다).

⚠ GradientExplainer 는 모델 출력이 `(N, n_outputs)` 2차원이라고 가정한다.
   우리 회귀 모델은 `(N,)` 을 반환하므로 그대로 넣으면
   `IndexError: too many indices for tensor of dimension 1` 이 난다 → _Squeeze2D 로 감싼다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
import torch

from . import config

logger = logging.getLogger(__name__)


class _Squeeze2D(torch.nn.Module):
    """(N,) 출력을 (N, 1) 로 되돌려 SHAP 의 2차원 가정을 맞춘다."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        return out.unsqueeze(-1) if out.dim() == 1 else out


def shap_importance(model, X_background: np.ndarray, X_explain: np.ndarray,
                    feature_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns: (피처별 중요도 표, 원시 shap 값 (N, L, F))."""
    wrapped = _Squeeze2D(model.to("cpu").eval()).eval()
    bg = torch.from_numpy(X_background)
    ex = torch.from_numpy(X_explain)

    with torch.backends.cudnn.flags(enabled=False):
        explainer = shap.GradientExplainer(wrapped, bg)
        sv = explainer.shap_values(ex)
    sv = sv[0] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    if sv.ndim == 4 and sv.shape[-1] == 1:      # (N, L, F, 1) → (N, L, F)
        sv = sv[..., 0]

    mean_abs = np.abs(sv).mean(axis=(0, 1))                  # 시간축·표본축 평균
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df["share_pct"] = 100 * df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    logger.info("SHAP 완료: 표본 %d, shap%s", len(X_explain), sv.shape)
    return df, sv


def time_profile(sv: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """시간축 프로파일 — 윈도우 안에서 '언제'가 중요했나 (최근 vs 과거)."""
    prof = np.abs(sv).mean(axis=0)                            # (L, F)
    return pd.DataFrame(prof, columns=feature_names,
                        index=pd.RangeIndex(-prof.shape[0] + 1, 1, name="lag_min"))
