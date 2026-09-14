"""칼만필터 — 빙결 위험확률을 예측분포에서 직접 낸다.

## KF 를 어디에 쓰나

앞서 원시 센서에 KF 를 붙였다가 persistence 에 −28.6% 로 졌다(docs/07). 그때는 **평활할 잠재상태가
없었다** — 출구온도는 제어되는 변수라 추정할 열화가 없었다.

여기서는 역할이 다르다. KF 는 **예측분포**를 준다:
  · 상태 = [수준, 기울기]  (local linear trend)
  · 6시간 앞으로 전파하면 평균과 **분산**이 같이 나온다
  · 경로를 모의추출해 `P(min over 6h < 0)` 를 **직접** 계산한다
분류기가 학습으로 확률을 얻는 것과 달리, KF 는 **모형에서 확률이 나온다.** 비교 가치가 있다.

## 계절은 기후값으로 뺀다

`t61` 은 계절성이 강하므로 훈련구간의 (월, 시각) 중앙값을 기후값으로 빼고, **잔차에만 KF** 를 건다.
예측할 때 기후값을 다시 더한다. 이렇게 하면 KF 가 계절을 재학습할 필요가 없고,
`season` 기준선(리프트 0.0~1.2)보다 나은지가 곧 **AR 구조의 기여**가 된다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config, data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

HORIZON = 6
N_SIM = 400


def climatology(y: pd.Series, tr: np.ndarray) -> pd.Series:
    """훈련구간 (월, 시각) 중앙값. 미래 정보 없음."""
    k = pd.MultiIndex.from_arrays([y.index.month, y.index.hour])
    c = pd.Series(y.to_numpy()[tr], index=k[tr]).groupby(level=[0, 1]).median()
    return pd.Series(k.map(c).to_numpy(dtype=float), index=y.index).fillna(float(y[tr].median()))


def run_kf(resid: np.ndarray, q_lvl: float, q_slp: float, r_obs: float):
    """local linear trend KF. 반환: 각 시점의 사후 상태·공분산."""
    n = len(resid)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([q_lvl, q_slp])
    x = np.array([resid[0] if np.isfinite(resid[0]) else 0.0, 0.0])
    P = np.eye(2) * 10.0
    xs = np.zeros((n, 2))
    Ps = np.zeros((n, 2, 2))
    for t in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        z = resid[t]
        if np.isfinite(z):
            S = float((H @ P @ H.T)[0, 0]) + r_obs
            K = (P @ H.T / S).ravel()
            x = x + K * (z - float((H @ x)[0]))
            P = P - np.outer(K, H @ P)
        xs[t], Ps[t] = x, P
    return xs, Ps


def forecast_prob(xs, Ps, clim_future: np.ndarray, q_lvl, q_slp, r_obs, rng) -> np.ndarray:
    """6시간 경로를 모의추출해 P(min < 0). clim_future: (n, H) 미래 기후값."""
    n = len(xs)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_lvl, q_slp])
    out = np.empty(n)
    L0 = np.linalg.cholesky(Q + 1e-12 * np.eye(2))
    for t in range(n):
        try:
            L = np.linalg.cholesky(Ps[t] + 1e-9 * np.eye(2))
        except np.linalg.LinAlgError:
            out[t] = np.nan
            continue
        st = xs[t] + (L @ rng.standard_normal((2, N_SIM))).T          # (N_SIM, 2)
        mins = np.full(N_SIM, np.inf)
        for h in range(HORIZON):
            st = st @ F.T + (L0 @ rng.standard_normal((2, N_SIM))).T
            obs = st[:, 0] + rng.standard_normal(N_SIM) * np.sqrt(r_obs) + clim_future[t, h]
            mins = np.minimum(mins, obs)
        out[t] = float(np.mean(mins < 0.0))
    return out


def main() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for train in config.TRAINS:
        f = data.build(train)
        X, y, tr, va, te, cols = data.split(f, HORIZON)
        t61 = X["t61"]
        clim = climatology(t61, np.asarray(tr))
        resid = (t61 - clim).to_numpy()

        # 미래 기후값 (예측 시점에 알 수 있는 정보다 — 달력만 필요)
        cf = np.column_stack([clim.shift(-h).to_numpy() for h in range(1, HORIZON + 1)])
        cf = np.where(np.isfinite(cf), cf, float(clim[tr].median()))

        # 잡음 모수는 검증구간 Brier 로 고른다 (시험구간은 안 본다)
        best = None
        for q_lvl in (0.05, 0.2, 1.0):
            for r_obs in (0.2, 1.0, 4.0):
                xs, Ps = run_kf(resid, q_lvl, q_lvl * 0.05, r_obs)
                vm = np.asarray(va)
                pv = forecast_prob(xs[vm], Ps[vm], cf[vm], q_lvl, q_lvl * 0.05, r_obs, rng)
                ev_va = np.asarray((y[va] < 0).astype(int))
                k = np.isfinite(pv)
                if k.sum() < 100:
                    continue
                b = brier_score_loss(ev_va[k], pv[k])
                if best is None or b < best[0]:
                    best = (b, q_lvl, r_obs, xs, Ps)
        if best is None:
            continue
        _, q_lvl, r_obs, xs, Ps = best
        tm = np.asarray(te)
        pt = forecast_prob(xs[tm], Ps[tm], cf[tm], q_lvl, q_lvl * 0.05, r_obs, rng)
        ev_te = np.asarray((y[te] < 0).astype(int))
        k = np.isfinite(pt)
        p, ev = pt[k], ev_te[k]
        base = float((y[tr] < 0).mean())
        r = {"계열": train, "모델": "KF", "q_lvl": q_lvl, "r_obs": r_obs, "n": int(k.sum()),
             "AUC": round(float(roc_auc_score(ev, p)), 3),
             "Brier": round(float(brier_score_loss(ev, p)), 5),
             "Brier개선%": round(100 * (1 - brier_score_loss(ev, p)
                                     / brier_score_loss(ev, np.full_like(p, base))), 1)}
        for frac in (0.01, 0.05):
            kk = max(int(frac * len(p)), 10)
            top = np.argsort(-p)[:kk]
            r[f"리프트{int(frac*100)}"] = (round(float(ev[top].mean() / ev.mean()), 1)
                                          if ev.mean() > 0 else np.nan)
        rows.append(r)
        log.info("[%s] KF q_lvl=%.2f r_obs=%.1f | AUC %.3f Brier개선 %.1f%% 리프트1%% %.1f",
                 train, q_lvl, r_obs, r["AUC"], r["Brier개선%"], r["리프트1"])
    out = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.OUTPUT_DIR / "kf_metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n=== KF 예측분포 기반 위험확률 (시험 2022-01 이후) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
