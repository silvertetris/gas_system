"""KF 변수 진단 — SHAP 의 KF 판(版).

⚠ **SHAP 은 KF 에 적용하지 않는다.** SHAP 은 "학습된 함수 f(입력)→출력"의 기여도를
   분해하는 도구인데, KF 는 학습된 사상이 아니라 **명시적 베이즈 갱신식**이다.
   기여도를 억지로 계산해도 해석이 안 된다.

   대신 KF 에는 같은 질문("어느 신호가 얼마나 중요/신뢰되나")에 **직접 답하는 고유량**이 있다:

   - **Kalman gain K**: 관측 하나가 상태를 얼마나 움직이나. K→1 이면 계기를 믿고
     K→0 이면 모델(예측)을 믿는다. 즉 **계기 신뢰도의 직접 측정치**다.
   - **innovation RMSE**: 그 신호를 상태모델이 얼마나 못 따라가나. 크면 잡음이 크거나
     동역학이 모델과 안 맞는다는 뜻 → PINN 에서 그 신호에 정제 효과를 기대하기 어렵다.
   - **NIS (normalized innovation squared)**: innovation²/S. 모델이 맞으면 자유도 1의
     χ² 를 따르므로 **평균 1 근처**여야 한다. 1보다 크게 벗어나면 Q/R 이 잘못 잡힌 것이다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def signal_diagnostics(Z: np.ndarray, res: dict[str, np.ndarray],
                       signals: list[str], scale: np.ndarray) -> pd.DataFrame:
    """신호별 KF 진단표."""
    inn = res["innovation"]
    rows = []
    for i, s in enumerate(signals):
        nis = res["nis"][:, i]
        rows.append({
            "signal": s,
            "kalman_gain_mean": float(np.nanmean(res["gain"][:, i])),
            "innovation_rmse_raw": float(np.sqrt(np.nanmean(inn[:, i] ** 2)) * scale[i]),
            "innovation_rmse_scaled": float(np.sqrt(np.nanmean(inn[:, i] ** 2))),
            "nis_mean": float(np.nanmean(nis)),
            "nis_over_1pct": float(100 * np.nanmean(nis > 3.84)),   # χ²(1) 95% 임계
        })
    df = pd.DataFrame(rows)
    # 신호 자체 변동폭 대비 예측오차 — 스케일 다른 신호를 공정 비교
    sd = np.nanstd(Z, axis=0)
    df["ninnov_vs_sd"] = (df["innovation_rmse_scaled"] / np.where(sd > 1e-9, sd, 1.0)).round(4)
    return df.sort_values("ninnov_vs_sd", ascending=False).reset_index(drop=True)


def outlier_flags(res: dict[str, np.ndarray], signals: list[str],
                  index, chi2_thresh: float = 3.84) -> pd.DataFrame:
    """NIS 기준 이상 시각 플래그 → PINN 학습에서 뺄 후보 구간."""
    nis = res["nis"]
    flag = np.nansum(nis > chi2_thresh, axis=1)
    return pd.DataFrame({"Time": index, "n_signals_flagged": flag,
                         "nis_max": np.nanmax(nis, axis=1),
                         "nis_mean": np.nanmean(nis, axis=1)})
