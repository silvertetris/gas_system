"""선형 가우시안 칼만 필터 (numpy 구현, 외부 의존 없음).

상태공간 (신호마다 독립, 신호 간 상관은 두지 않는다):

  local_level :  x = [level],          level_{t+1} = level_t + w
  local_trend :  x = [level, slope],   level_{t+1} = level_t + slope_t·dt + w1
                                       slope_{t+1} = slope_t + w2
  관측        :  z_t = level_t + v,    v ~ N(0, R)

**왜 신호별 독립인가**: 신호 간 결합은 물리식이 규정하는 것이고 그건 PINN 의 몫이다(config 참고).
여기서 상관을 임의로 넣으면 PINN 이 추정해야 할 구조를 KF 가 먼저 가정해 버린다.

결측(NaN) 관측은 **갱신 단계를 건너뛰고 예측만** 수행한다 — 칼만필터가 dropout 을
자연스럽게 다루는 방식이고, prex 의 "보간 금지" 원칙과도 맞는다.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_matrices(model_type: str, dt: float, q_level: float, q_slope: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(F 전이행렬, Q 프로세스잡음, H 관측행렬) — 신호 하나에 대한 것."""
    if model_type == "local_level":
        F = np.array([[1.0]])
        Q = np.array([[q_level]])
        H = np.array([[1.0]])
    else:  # local_trend
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[q_level, 0.0], [0.0, q_slope]])
        H = np.array([[1.0, 0.0]])
    return F, Q, H


def run_filter(Z: np.ndarray, params: dict, dt: float = 1.0
               ) -> dict[str, np.ndarray]:
    """모든 신호에 KF 를 돌린다.

    Args:
        Z: (T, S) 관측. NaN 허용.
    Returns dict:
        filtered  (T, S)  갱신 후 상태 추정 = 정제된 신호      → PINN 입력
        predicted (T, S)  1스텝 앞 예측(갱신 전)               → 성능 평가 기준
        innovation(T, S)  z - 예측                              → 이상 플래그 재료
        nis       (T, S)  정규화 innovation 제곱 (χ² 통계)      → 이상 판정
        gain      (T, S)  칼만 이득 (관측을 얼마나 믿는가)
        var       (T, S)  상태 분산
    """
    T, S = Z.shape
    F, Q, H = build_matrices(params["model_type"], dt,
                             params["q_level"], params.get("q_slope", 0.0))
    k = F.shape[0]
    R = float(params["r_scale"])

    out = {key: np.full((T, S), np.nan) for key in
           ("filtered", "predicted", "innovation", "nis", "gain", "var")}

    for s in range(S):
        z = Z[:, s]
        first = np.flatnonzero(~np.isnan(z))
        if len(first) == 0:
            continue
        x = np.zeros((k, 1))
        x[0, 0] = z[first[0]]
        P = np.eye(k) * 1.0

        for t in range(T):
            # --- 예측 ---
            x = F @ x
            P = F @ P @ F.T + Q
            z_pred = float((H @ x)[0, 0])
            S_inn = float((H @ P @ H.T)[0, 0] + R)
            out["predicted"][t, s] = z_pred
            out["var"][t, s] = float(P[0, 0])

            # --- 갱신 (관측이 있을 때만) ---
            if not np.isnan(z[t]):
                nu = z[t] - z_pred
                K = (P @ H.T) / S_inn
                x = x + K * nu
                P = P - K @ H @ P
                out["innovation"][t, s] = nu
                out["nis"][t, s] = nu * nu / S_inn
                out["gain"][t, s] = float(K[0, 0])
            out["filtered"][t, s] = float((H @ x)[0, 0])
    return out


def one_step_rmse(Z: np.ndarray, pred: np.ndarray, scale: np.ndarray | None = None
                  ) -> tuple[float, np.ndarray]:
    """1스텝 앞 예측 RMSE. Returns (신호 평균, 신호별 배열).

    **왜 이 지표인가**: KF 는 정제기라서 '자기 자신을 얼마나 잘 복원하나'로 평가하면
    필터링을 아예 안 하는 쪽이 항상 이긴다(R→0). 반면 **다음 관측을 얼마나 잘 맞히나**는
    상태 모델의 품질을 정직하게 잰다 — 과평활하면 전이를 놓쳐 벌을 받고,
    과소평활하면 잡음을 따라가 벌을 받는다.

    ⚠ **`scale` 은 기본값 None 으로 두고 표준화된 입력을 넣을 것.**
    scale 을 주면 원 단위로 환산해 평균내는데, 신호가 이질적이면 그 평균이 무의미하다 —
    실측(2026-09-11) PI-D2P(mmHg, σ18)·TI33P(℃, σ9.7)·PI21X(MPa, σ0.31)를 원 단위로
    단순평균하니 PI-D2P 하나가 지표를 지배했다. 표준화 공간(σ=1)에서 재야 공정하다.
    """
    err = Z - pred
    per = np.sqrt(np.nanmean(err ** 2, axis=0))
    if scale is not None:
        per = per * scale
    return float(np.nanmean(per)), per
