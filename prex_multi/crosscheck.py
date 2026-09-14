"""(H1) 수조 열수지로 (H5) 합류점 식을 교차검증한다 (docs §18-5).

**왜 이 검정이 결정적인가**

한 계열에서 히터 하나가 `flow`, 다른 하나가 `isolated` 인 시점을 고르면 (H5)가 아주 단순해진다:
    ΔH = f·Δ           (차단된 히터는 f=0)   →   f = ΔH/Δ       ← U·A 도 ε 도 필요 없다
그 히터가 수조에서 빼가는 열은
    Q_gas = f·m_tot·c_p·Δ = m_tot·c_p·ΔH
즉 **Δ(그 히터의 가스 승온폭)가 아니라 ΔH(헤더 승온폭)에 비례**한다. 버너 OFF 면
    −C_eff·dT_bath/dt = m_tot·c_p·**ΔH** + Q_loss          … (H5 가 맞을 때)

바이패스가 없다면(구 모델) 그 히터를 지나는 것이 전량이므로 f=1, Q_gas = m_tot·c_p·Δ 이고
    −C_eff·dT_bath/dt = m_tot·c_p·**Δ** + Q_loss           … (H5 가 틀릴 때)

두 예측은 **서로 다른 설명변수**를 지목한다. `dT_bath/dt` 는 (H3)·(H5) 어디에도 안 쓰인 신호다.

⚠ **단순 R² 비교는 편향된다.** `Δ = T_out − T_in` 의 `T_out` 은 수조온도에 끌려가고(열교환),
수조온도가 높으면 손실도 커져 냉각이 빨라진다. 즉 `Δ` 는 `T_bath` 를 통한 **가짜 경로**로
냉각률과 상관된다. `ΔH` 에는 그 경로가 없다. 실제로 단순 R² 로는 `Δ` 가 4건 중 3건을 이겼다.
→ **`T_bath` 를 통제한 증분 R²** 로 판정한다:
      M0: y ~ T_bath            (기저)
      M1: y ~ T_bath + Δ        (바이패스 없음이 주장하는 추가 설명력)
      M2: y ~ T_bath + ΔH       (H5 가 주장하는 추가 설명력)
      M3: y ~ T_bath + Δ + ΔH   (둘 다)
  `ΔH` 가 `Δ` 위에 얹어도 설명력을 더한다면(M3 > M1) (H5)를 지지하는 증거다.

⚠ 양자화: dT_bath/dt 는 인접 차분이 아니라 **OFF 런 내부 최소제곱 기울기**로 구한다(§15 교훈).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SETTLE_MIN = 10      # OFF 직후 과도구간 버림
MIN_RUN_LEN = 20     # 런 최소 길이 [분]
MIN_BATH_C = 20.0    # 계측 드롭아웃 제외
MIN_DELTA_C = 1.0    # Δ, ΔH 하한 [℃]
TRIM = 0.05


def burner_states(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """[구] DI 재구성 함수 → **전처리로 이관됐다**(`prex_multi/trend.py::add_burner_state`).

    호환용 얇은 래퍼로만 남긴다. DI 3함정 보정은 전처리에서 한 번만 한다 — 분석 모듈에서
    다시 구현하면 §14-4 함정을 다시 밟는다.
    """
    g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet",
                        columns=["Time"] + [f"burner_{u}" for u in config.UNITS])
    g = g.set_index("Time")
    return {u: g[f"burner_{u}"].reindex(index) for u in config.UNITS}


def off_runs(g: pd.DataFrame, burner: pd.Series, unit: str, other: str,
             train: str) -> pd.DataFrame:
    """`unit`=flow, `other`=isolated, 버너 OFF 인 연속 런의 요약."""
    bath = g[f"TI-D2{unit}"]
    t_in = g[config.TRAIN_INLET[train]]
    delta = g[f"TI33{unit}"] - t_in
    dh = g[f"t_hdr_{train}"] - t_in

    ok = ((burner == 0) & (g[f"regime_{unit}"] == "flow")
          & (g[f"regime_{other}"] == "isolated")
          & bath.notna() & (bath > MIN_BATH_C)
          & (delta > MIN_DELTA_C) & (dh > MIN_DELTA_C))
    if ok.sum() < 1000:
        return pd.DataFrame()

    run = ((ok != ok.shift()) | (g["segment_id"] != g["segment_id"].shift())).cumsum()
    f = pd.DataFrame({"run": run[ok], "T": bath[ok].to_numpy(),
                      "d": delta[ok].to_numpy(), "dh": dh[ok].to_numpy()})
    f = f[f.groupby("run").cumcount() >= SETTLE_MIN]
    if f.empty:
        return pd.DataFrame()
    f["t"] = f.groupby("run").cumcount().to_numpy() * 60.0

    grp = f.groupby("run")
    n, tb, Tb = grp.size(), grp["t"].mean(), grp["T"].mean()
    stt = grp["t"].apply(lambda s: float(((s - s.mean()) ** 2).sum()))
    stT = f.assign(tT=f["t"] * f["T"]).groupby("run")["tT"].sum() - n * tb * Tb
    out = pd.DataFrame({"L": n, "slope": stT / stt.replace(0.0, np.nan),
                        "delta": grp["d"].median(), "dH": grp["dh"].median(),
                        "T_bath": Tb})
    return out[(out["L"] >= MIN_RUN_LEN) & out["slope"].notna()
               & (out["slope"] < 0)].reset_index(drop=True)   # 냉각 중인 런만


def mad_keep(*arrs: np.ndarray, k: float = 4.0) -> np.ndarray:
    """MAD 기준 이상치 마스크. ⚠ 모델 비교 전에 **한 번만** 적용해 표본을 고정해야 한다."""
    keep = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        med = np.median(a)
        mad = np.median(np.abs(a - med)) * 1.4826
        if mad > 0:
            keep &= np.abs(a - med) <= k * mad
    return keep


def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """단순 OLS 결정계수. 절사 없음 — 표본은 호출자가 이미 고정해 둔 것을 쓴다.

    ⚠ 과거 버그: 모델마다 잔차 절사를 따로 하면 **R² 를 비교할 수 없다**(표본이 달라진다).
      실제로 그 버그로 R² 가 −2.195 ~ 0.943 사이를 널뛰었다. 절대 되돌리지 말 것.
    """
    X = X[:, None] if X.ndim == 1 else X
    A = np.c_[X, np.ones(len(X))]
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ beta
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum(r ** 2) / ss) if ss > 0 else np.nan


def main() -> None:
    g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()
    bn = burner_states(g.index)
    log.info("대상 %s", g.shape)

    rows = []
    for train, units in config.TRAIN_UNITS.items():
        for a, b in [(0, 1), (1, 0)]:
            u, o = units[a], units[b]
            R = off_runs(g, bn[u], u, o, train)
            if len(R) < 50:
                log.info("[%s] %s=flow, %s=isolated : 런 %d개 — 표본 부족", train, u, o, len(R))
                continue
            y = (-R["slope"]).to_numpy(float)          # 냉각률 [K/s], 양수
            Tb = R["T_bath"].to_numpy(float)
            dd = R["delta"].to_numpy(float)
            dh = R["dH"].to_numpy(float)
            keep = mad_keep(y, Tb, dd, dh)          # ★ 한 번만, 네 모델 공통 표본
            y, Tb, dd, dh = y[keep], Tb[keep], dd[keep], dh[keep]
            if len(y) < 80:
                continue
            M0 = {"r2": ols_r2(Tb, y)}
            M1 = {"r2": ols_r2(np.c_[Tb, dd], y)}
            M2 = {"r2": ols_r2(np.c_[Tb, dh], y)}
            M3 = {"r2": ols_r2(np.c_[Tb, dd, dh], y)}
            rows.append({
                "계열": train, "히터": u, "차단": o, "런수": len(y),
                "corr(Δ,ΔH)": round(float(np.corrcoef(dd, dh)[0, 1]), 3),
                "R2 M0(T_bath)": round(M0["r2"], 3),
                "R2 M1(+Δ)": round(M1["r2"], 3),
                "R2 M2(+ΔH)": round(M2["r2"], 3),
                "R2 M3(+둘다)": round(M3["r2"], 3),
                "ΔH 증분(M3−M1)": round(M3["r2"] - M1["r2"], 3),
                "Δ 증분(M3−M2)": round(M3["r2"] - M2["r2"], 3),
            })
            log.info("[%s] %s=flow, %s=iso : 런 %d | R² M0 %.3f → +Δ %.3f, +ΔH %.3f, 둘다 %.3f"
                     " | ΔH증분 %+.3f, Δ증분 %+.3f",
                     train, u, o, len(R), M0["r2"], M1["r2"], M2["r2"], M3["r2"],
                     M3["r2"] - M1["r2"], M3["r2"] - M2["r2"])
    out = pd.DataFrame(rows)
    out.to_csv(config.OUTPUT_DIR / "crosscheck_h1_h5.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== (H1) 수조 열수지로 본 (H5) 검증 — T_bath 통제 증분 R² ===")
    print("  H5 가 옳다면 ΔH 증분(M3−M1) > 0 이어야 한다 (Δ 위에 얹어도 설명력을 더함)")
    print("  바이패스가 없다면 ΔH 증분 ≈ 0, Δ 증분(M3−M2) > 0 이어야 한다")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
