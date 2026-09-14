"""단순 평활기 베이스라인 — AE 의 PCA 에 해당.

KF 가 이들을 못 이기면 상태공간 모델을 쓸 이유가 없다.
모두 **1스텝 앞 예측** 으로 비교한다(model.one_step_rmse 와 같은 기준).
"""
from __future__ import annotations

import numpy as np


def persistence(Z: np.ndarray) -> np.ndarray:
    """ẑ_t = z_{t-1}. 필터를 전혀 안 쓰는 랜덤워크 가정."""
    p = np.full_like(Z, np.nan)
    p[1:] = Z[:-1]
    return p


def ewma(Z: np.ndarray, alpha: float) -> np.ndarray:
    """지수가중이동평균의 1스텝 예측. NaN 은 상태를 유지한 채 건너뛴다."""
    T, S = Z.shape
    pred = np.full_like(Z, np.nan)
    state = np.full(S, np.nan)
    for t in range(T):
        pred[t] = state
        z = Z[t]
        m = ~np.isnan(z)
        new = np.where(np.isnan(state), z, alpha * z + (1 - alpha) * state)
        state = np.where(m, new, state)
    return pred


def moving_average(Z: np.ndarray, window: int) -> np.ndarray:
    """직전 window 개 평균으로 다음 값을 예측."""
    T, S = Z.shape
    pred = np.full_like(Z, np.nan)
    for t in range(1, T):
        lo = max(0, t - window)
        seg = Z[lo:t]
        if len(seg):
            with np.errstate(invalid="ignore"):
                pred[t] = np.nanmean(seg, axis=0)
    return pred
