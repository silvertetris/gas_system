"""EKF 정제 — 헤더→공급온도 **비선형 물리식**과 관측의 일관성을 본다.

## 역할 (예측기가 아니다)

AE·KF·EKF 는 **정제 도구**다. 출력은 PINN 의 입력으로만 쓰고, 평가는 "넣었을 때 PINN 이
좋아지는가"로 한다. 셋은 보는 것이 다르다:

| 도구 | 보는 것 | 산출 |
|---|---|---|
| KF  (`risk/refine.py`) | 태그 하나의 선형 흐름 | 평활값, 혁신 |
| AE  (`risk/refine.py`) | 다변수 조합의 정상성 (모드별) | 재구성 오차 |
| **EKF (여기)** | (Q5)+(Q6) 비선형 물리와 관측의 정합 | 물리 일관 상태, **물리 이탈 혁신**, 유량 결측 추정 |

## 상태공간

    상태 x = [T_hdr, ΔP, T_in, log m]
    전이   T_hdr·ΔP·T_in : 랜덤워크
           log m         : AR(1) → 학습평균 (열수지 유량이 장시간 비면 분산이 발산하지 않게)
    관측   z = [T_hdr, ΔP, T_in, log m(있을 때만), T_61]
           h(x) = [T_hdr, ΔP, T_in, log m, f(x)]
           f(x) = T_g + (T_hdr − μ_JT·ΔP − T_g)·exp(−N),  N = U·A·e^{−log m}/c_p + N₀,  T_g = a + b·T_in

**T_61 관측이 log m 을 야코비안으로 끌어당긴다** — 열수지 유량이 없는 시각(M 32%·Z 69%)에도
물리식이 유량을 추정한다. 이것이 KF·AE 에 없는 EKF 고유의 정제 결과다.

야코비안 (해석해):
    ∂f/∂T_hdr = e,   ∂f/∂ΔP = −μ·e,   ∂f/∂T_in = b(1−e),   ∂f/∂log m = (T_hdr−μΔP−T_g)·e·(N−N₀)
    (e = exp(−N))

## 누수 방지

  · 물리 모수(μ_JT, U·A, N₀, a, b)는 **학습구간에서만** 비선형 최소제곱으로 따로 적합한다.
    PINN 이 학습한 값을 쓰지 않는다(순환 방지).
  · 잡음 Q·R 도 학습구간에서만 적률법으로 추정한다: 랜덤워크+백색잡음에서
    var(Δx) = Q + 2R,  cov(Δx_t, Δx_{t−1}) = −R.
  · 필터는 **전방향(인과)** 이다. 시각 t 의 산출은 t 까지의 관측만 쓴다.
  · 3시간 넘는 공백에서 리셋한다 (KF 의 segment 리셋과 같은 원칙).

## 해상도 — 1시간

KF·AE 는 버너 사이클(ON 8~18분)과 운전 모드 때문에 1분에서 한다. EKF 가 보는 헤더→공급
구간은 매설배관·지중 열용량이 지배해 시정수가 수 시간이다. 1시간으로 충분하고,
1분 원본(TI61x·PI61x·PI21X 는 원본 CSV 에만 있다)을 780만 행 순차 필터링할 이유가 없다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config, data

logger = logging.getLogger(__name__)

STATE = ("hdr", "dp", "t_in", "log_m")
RESET_GAP_H = 3
P0 = np.array([4.0, 4.0, 4.0, 1.0])          # 리셋 직후 상태 분산


def _train_mask(index: pd.DatetimeIndex, fit_end=None) -> np.ndarray:
    """적합 구간. `fit_end` 가 없으면 기존 분할(TRAIN_END − 퍼지), 있으면 그 시각 이전.

    Optuna 의 TimeSeriesSplit fold 마다 **그 fold 학습 구간 끝**을 넘겨 재적합한다 — 그래야
    초기 fold(검증창이 2019 이전)에 미래 정보가 새지 않는다.
    """
    if fit_end is None:
        end = pd.Timestamp(config.TRAIN_END) - pd.Timedelta(hours=config.PURGE_H + config.HORIZON_H)
    else:
        end = pd.Timestamp(fit_end)
    return np.asarray((index >= pd.Timestamp(config.VALID_FROM)) & (index < end))


def physics(theta, hdr, dp, tin, logm):
    mu, ua, n0, a, b = theta
    tg = a + b * tin
    n = ua * np.exp(-logm) / config.CP_GAS_J_KGK + n0
    e = np.exp(-n)
    return tg + (hdr - mu * dp - tg) * e, e, n, tg


def fit_params(f: pd.DataFrame, train: str, fit_end=None) -> np.ndarray:
    """학습구간·유량관측 행에서 (Q5)+(Q6) 모수를 최소제곱 적합."""
    tr = _train_mask(f.index, fit_end) & (f["m_obs"].to_numpy() > 0.5)
    d = f.loc[tr, ["t61", "hdr", "dp", "t_in", "log_m"]].dropna()
    x0 = [config.MU_JT_INIT, config.UA_PIPE_INIT_KW[train] * 1e3, config.N0_INIT, 0.0, 1.0]
    lo = [config.MU_JT_MIN, 10.0, 0.0, -20.0, 0.0]
    hi = [config.MU_JT_MAX, 1e6, 5.0, 20.0, 2.0]

    def res(th):
        return physics(th, d["hdr"].to_numpy(), d["dp"].to_numpy(), d["t_in"].to_numpy(),
                       d["log_m"].to_numpy())[0] - d["t61"].to_numpy()

    r = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0)
    e = res(r.x)
    rmse = float(np.sqrt(np.mean(e ** 2)))
    # ⚠ 관측잡음은 **강건 분산**으로 잡는다. 1차에 RMSE² 를 쓰니 꼬리가 두꺼운 M 잔차 때문에
    #   R_T61 이 과대(34.9)해져 NIS 중앙이 0.09·0.21 (기대 ≈0.45) 로 나왔다. R 이 크면 T_61 이
    #   log m 을 약하게 끌어 결측 시각 유량 추정이 무뎌진다.
    r_robust = float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    logger.info("[%s] EKF 물리모수 (학습 %d행): μ_JT %.3f  U·A %.2f kW/K  N₀ %.3f  "
                "T_g=%.2f%+.3f·T_in  잔차RMSE %.2f℃",
                train, len(d), r.x[0], r.x[1] / 1e3, r.x[2], r.x[3], r.x[4], rmse)
    logger.info("[%s] R_T61 강건분산 %.3f (RMSE² 였다면 %.3f)", train, r_robust, rmse ** 2)
    return r.x, r_robust


def _moments(s: pd.Series, tr: np.ndarray) -> tuple[float, float]:
    """랜덤워크+백색잡음 적률법 → (Q, R). 음수는 바닥값으로."""
    v = s.where(tr).to_numpy(np.float64)
    dx = np.diff(v)
    ok = np.isfinite(dx[1:]) & np.isfinite(dx[:-1])
    var = float(np.nanvar(dx))
    cov = float(np.mean((dx[1:][ok] - np.nanmean(dx)) * (dx[:-1][ok] - np.nanmean(dx)))) if ok.any() else 0.0
    r = max(-cov, 1e-4)
    q = max(var - 2 * r, 1e-4)
    return q, r


def run(train: str, fit_end=None, out_path=None, params_path=None) -> pd.DataFrame:
    # ⚠ 기반 데이터만 읽는다. `data.build` 는 기본 정제(KF·AE·EKF)를 붙이므로 EKF 자신의
    #   이전 산출물을 읽는 **순환 의존**이 생긴다(파일이 없으면 실패, 있으면 불필요한 결합).
    f = data.build_base(train).copy()
    f = f[f.index >= pd.Timestamp(config.VALID_FROM)]
    tr = _train_mask(f.index, fit_end)
    theta, r_t61 = fit_params(f, train, fit_end)
    mu, ua, n0, a, b = theta

    q = np.zeros(4); r = np.zeros(4)
    for i, c in enumerate(STATE):
        q[i], r[i] = _moments(f[c], tr)
    lm_tr = f.loc[tr & (f["m_obs"].to_numpy() > 0.5), "log_m"].dropna()
    lm_mean = float(lm_tr.mean())
    ac = lm_tr.autocorr(1)
    phi = float(np.clip(ac if np.isfinite(ac) else 0.98, 0.5, 0.999))
    q[3] = max(float(lm_tr.var()) * (1 - phi ** 2), 1e-4)       # 정상 AR(1) 분산과 정합
    logger.info("[%s] Q %s  R %s  R_T61 %.3f  φ(log m) %.3f",
                train, np.round(q, 4), np.round(r, 4), r_t61, phi)

    # copy=True — pandas copy-on-write 에서 to_numpy 는 읽기전용 뷰를 줄 수 있다(아래에서 덮어쓴다)
    Z = f[["hdr", "dp", "t_in", "log_m", "t61"]].to_numpy(np.float64, copy=True)
    Z[:, 3] = np.where(f["m_obs"].to_numpy() > 0.5, Z[:, 3], np.nan)
    Rfull = np.array([r[0], r[1], r[2], r[3], r_t61])
    t_m = f.index.values.astype("datetime64[m]").astype(np.int64)     # 분 (해상도 무관)

    n = len(f)
    out = np.full((n, 7), np.nan)       # hdr dp tin logm logm_sd innov_t61 nis_t61
    x = None; P = None; F = np.diag([1, 1, 1, phi]); Q = np.diag(q)
    for k in range(n):
        z = Z[k]
        if x is None or (k > 0 and t_m[k] - t_m[k - 1] > RESET_GAP_H * 60):
            if not np.isfinite(z[:3]).all():
                x = None
                continue
            x = np.array([z[0], z[1], z[2], z[3] if np.isfinite(z[3]) else lm_mean])
            P = np.diag(P0)
        else:
            x = F @ x + np.array([0, 0, 0, (1 - phi) * lm_mean])
            P = F @ P @ F.T + Q
        # 관측 모형과 야코비안
        pred, e, nn, tg = physics(theta, x[0], x[1], x[2], x[3])
        h = np.array([x[0], x[1], x[2], x[3], pred])
        H = np.zeros((5, 4)); H[:4, :4] = np.eye(4)
        H[4] = [e, -mu * e, b * (1 - e), (x[0] - mu * x[1] - tg) * e * (nn - n0)]
        m = np.isfinite(z)
        if m[4]:
            s = H[4] @ P @ H[4] + r_t61
            out[k, 5] = z[4] - pred                                   # 사전 혁신 (인과)
            out[k, 6] = (z[4] - pred) ** 2 / s
        if m.any():
            Hm, Rm = H[m], np.diag(Rfull[m])
            S = Hm @ P @ Hm.T + Rm
            K = P @ Hm.T @ np.linalg.solve(S, np.eye(len(S)))
            x = x + K @ (z[m] - h[m])
            P = (np.eye(4) - K @ Hm) @ P
            P = 0.5 * (P + P.T)
        out[k, :4] = x
        out[k, 4] = np.sqrt(max(P[3, 3], 0.0))

    res = pd.DataFrame(out, index=f.index, columns=[
        "ekf_hdr", "ekf_dp", "ekf_tin", "ekf_logm", "ekf_logm_sd", "innov_ekf_t61", "nis_ekf_t61"])
    obs = f["m_obs"].to_numpy() > 0.5
    both = obs & np.isfinite(res["ekf_logm"]).to_numpy() & np.isfinite(f["log_m"]).to_numpy()
    logger.info("[%s] EKF 유량 — 관측시각 추정오차 RMSE(log) %.3f | 결측시각 %d행 채움 | "
                "사전혁신 RMSE %.2f℃ | NIS 중앙 %.2f (≈0.45 면 잡음 정합)",
                train, float(np.sqrt(np.nanmean((res["ekf_logm"][both] - f["log_m"][both]) ** 2))),
                int((~obs & np.isfinite(res["ekf_logm"]).to_numpy()).sum()),
                float(np.sqrt(np.nanmean(res["innov_ekf_t61"] ** 2))),
                float(np.nanmedian(res["nis_ekf_t61"])))
    pd.DataFrame([dict(zip(["μ_JT", "UA_W_per_K", "N0", "a", "b"], theta),
                       R_T61=r_t61, phi_logm=phi, 계열=train, fit_end=str(fit_end))]).to_csv(
        params_path or config.OUTPUT_DIR / f"ekf_params_{train}.csv", index=False, encoding="utf-8-sig")
    res.to_parquet(out_path or config.OUTPUT_DIR / f"ekf_{train}.parquet")
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for t in config.TRAINS:
        run(t)


if __name__ == "__main__":
    main()
