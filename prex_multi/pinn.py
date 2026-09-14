"""물리제약 공동추정 — `U·A(t)` 를 분리해낸다.

## 왜 이게 남은 길인가

지금까지의 검정은 **원시 관측량**(`eps`·`ntu`·`beta`·`drive`·`dT`·`duty`)에서 정비 회복을 찾았고
전부 실패했다(docs/07). 그런데 거기엔 빈 곳이 있다:

    NTU = U·A / (m·c_p)

유량 `m` 이 설계의 13~100% 로 흔들린다. `U·A` 가 천천히 떨어져도 **`m` 의 변동이 덮는다.**
"`NTU` 에 추세가 없다"가 "`U·A` 에 추세가 없다"를 뜻하지 않는다. → `U·A` 를 분리해야 한다.

## 식별 원리 — 이것이 핵심

물리식 두 개를 시점마다 세우면:

    (R1)  C_eff·dT_bath/dt = fire·s·Q_b − m·c_p·Δ − Q_loss      →  m(t) 가 대수적으로 결정
    (R2)  ε = 1 − exp(−U·A/(m·c_p))                              →  U·A(t) = NTU·m·c_p

**모수 3개(`fire`, `C_eff`, `Q_loss`)만 정하면 상태 `m(t)`·`U·A(t)` 는 바로 나온다.**
그럼 모수는 무엇이 정하나 — **`U·A` 는 설비 물성이라 천천히만 변한다**는 물리 제약이다.
모수가 틀리면 `U·A(t)` 가 **부하를 따라 고주파로 요동친다.**

    목적함수 = (하루 안 U·A 변동) + λ·|corr(U·A, 부하)|
    제약     = 0 < fire ≤ 1,  C_eff ≥ 현열,  Q_loss ≥ 0,  m > 0,  U·A > 0

이것이 앞선 시도와 다른 점이다. 전에는 모수를 **구간마다 따로** 추정해 오차를 전파시켰고
(`ua>0`·`Σf≤1` 같은) 필터가 선택 편향을 만들었다. 여기서는 모수가 **전 구간 공유 스칼라 3개**뿐이라
편향이 들어올 자리가 없다.

## 사전에 정한 판정 기준 (모델을 본 뒤에 바꾸지 않는다)

  관문1  물리적 타당성 — `fire ≤ 1`, `C_eff ≥ 현열`, `Q_loss ≥ 0`, `U·A` 가 설계의 0.2~2배
  관문2  분리 성공    — `U·A` 가 부하와 무상관(|ρ| < 0.3). 아니면 여전히 유량을 보고 있는 것
  관문3  정비 회복    — `U·A(t)` 가 정비 7건에서 회복. **대조군 대비 단측 순위검정 p<0.05**

셋 다 통과해야 트랙2(Degradation)가 성립한다. 하나라도 실패하면 그 지점에서 멈춘다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import calorimetry, config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

WINDOW_MIN = 60
MIN_COVERAGE = 0.9
MIN_DELTA_C = 5.0
MIN_SLOPE_SAMPLES = 40
LAMBDA_LOAD = 2.0          # 부하 상관 벌점 가중
MAINTENANCE = pd.to_datetime(["2013-03-28", "2015-03-16", "2017-03-31", "2019-03-21",
                              "2021-04-02", "2022-10-21", "2024-10-29"])


def build_windows(g: pd.DataFrame, u: str) -> pd.DataFrame:
    """창 단위 관측: 버너 가동률·가스 승온폭·NTU·수조 승온율."""
    key = g.index.floor(f"{WINDOW_MIN}min")
    t_in = g[config.TRAIN_INLET[config.TRAIN[u]]]
    bath = g[f"TI-D2{u}"]

    f = pd.DataFrame({"k": key, "y": bath.to_numpy()}).dropna()
    f["t"] = f.groupby("k").cumcount().to_numpy() * 60.0
    gr = f.groupby("k")
    n = gr.size()
    st, sy = gr["t"].sum(), gr["y"].sum()
    stt = (f["t"] ** 2).groupby(f["k"]).sum()
    sty = (f["t"] * f["y"]).groupby(f["k"]).sum()
    rate = ((sty - st * sy / n) / (stt - st ** 2 / n).replace(0.0, np.nan)).where(n >= MIN_SLOPE_SAMPLES)

    W = pd.DataFrame({
        "s": g[f"burner_{u}"].groupby(key).mean(),
        "d": (g[f"TI33{u}"] - t_in).groupby(key).median(),
        "ntu": g[f"ntu_{u}"].groupby(key).median(),
        "rate": rate,
        "cov": bath.notna().groupby(key).mean(),
        "reg": g[f"regime_{u}"].groupby(key).agg(
            lambda s: s.iloc[0] if s.notna().all() and s.nunique() == 1 else None),
    })
    ok = ((W["reg"] == "flow") & (W["d"] > MIN_DELTA_C) & (W["cov"] > MIN_COVERAGE)
          & W["rate"].notna() & W["ntu"].notna() & (W["ntu"] > 0))
    return W[ok]


def states(W: pd.DataFrame, u: str, fire: float, c_eff: float, q_loss: float):
    """모수가 주어지면 상태는 대수적으로 결정된다."""
    qb = config.design_duty_w(u) / calorimetry.BURNER_EFF
    q_gas = fire * W["s"].to_numpy() * qb - q_loss - c_eff * W["rate"].to_numpy()
    m_cp = q_gas / W["d"].to_numpy()                       # m·c_p [W/K]
    ua = W["ntu"].to_numpy() * m_cp
    return m_cp, ua


def objective(W: pd.DataFrame, u: str, p: np.ndarray) -> float:
    """U·A 가 물성답게 '느리게' 나오도록 모수를 고른다."""
    fire, c_eff, q_loss = p
    m_cp, ua = states(W, u, fire, c_eff, q_loss)
    ok = np.isfinite(ua) & (ua > 0) & (m_cp > 0)
    if ok.mean() < 0.5:
        return 1e6                                          # 절반 이상이 비물리면 기각
    lu = np.log(ua[ok])
    idx = W.index[ok]
    day = pd.Series(lu, index=idx).groupby(idx.normalize())
    within = float(day.std().median())                      # 하루 안 변동 (작아야)
    load = W["s"].to_numpy()[ok]
    rho = abs(float(pd.Series(lu).corr(pd.Series(load), method="spearman")))
    return within + LAMBDA_LOAD * rho + 10.0 * (1.0 - ok.mean())


def fit(W: pd.DataFrame, u: str) -> dict:
    """3-모수 격자 + 국소 정련. 신경망이 필요 없다 — 모수가 셋뿐이다."""
    sens = config.scaled_mw_cw_j_per_k(u) if u != "P" else config.P_MW_CW_J_PER_K
    duty = config.design_duty_w(u)
    best = (1e9, None)
    for fire in np.linspace(0.05, 1.0, 20):
        for ce_mult in np.linspace(1.0, 12.0, 23):          # C_eff ≥ 현열
            for ql_frac in np.linspace(0.0, 0.06, 13):      # Q_loss = 설계열량의 0~6%
                p = np.array([fire, ce_mult * sens, ql_frac * duty])
                v = objective(W, u, p)
                if v < best[0]:
                    best = (v, p)
    v, p = best
    fire, c_eff, q_loss = p
    m_cp, ua = states(W, u, fire, c_eff, q_loss)
    ok = np.isfinite(ua) & (ua > 0) & (m_cp > 0)
    des = config.P_UA_W_PER_K if u == "P" else config.scaled_ua_w_per_k(u)
    return {"fire": fire, "C_eff_MJK": c_eff / 1e6, "C_eff_배": c_eff / sens,
            "Q_loss_kW": q_loss / 1e3, "Q_loss_%": 100 * q_loss / duty,
            "obj": v, "유효비율": float(ok.mean()),
            "UA_중앙_kWK": float(np.median(ua[ok])) / 1e3, "설계UA": des / 1e3,
            "설계비": float(np.median(ua[ok])) / des,
            "m설계비": float(np.median(m_cp[ok] / config.CP_GAS_J_KGK)) / config.design_mgas_kg_s(u),
            "UA_부하상관": float(pd.Series(np.log(ua[ok])).corr(
                pd.Series(W["s"].to_numpy()[ok]), method="spearman")),
            "_ua": pd.Series(ua[ok], index=W.index[ok])}


def main() -> None:
    g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()
    rows, series = [], {}
    for u in config.UNITS:
        W = build_windows(g, u)
        if len(W) < 2000:
            log.info("[%s] 창 %d개 — 부족", u, len(W))
            continue
        r = fit(W, u)
        series[u] = r.pop("_ua")
        r["창수"] = len(W)
        r["히터"] = u
        rows.append(r)
        log.info("[%s] 창 %d | fire=%.2f C_eff=%.0f MJ/K(현열×%.1f) Q_loss=%.1f kW(%.1f%%) "
                 "| U·A 중앙 %.1f kW/K (설계비 %.2f) | m/설계 %.2f | UA-부하상관 %+.2f",
                 u, len(W), r["fire"], r["C_eff_MJK"], r["C_eff_배"], r["Q_loss_kW"],
                 r["Q_loss_%"], r["UA_중앙_kWK"], r["설계비"], r["m설계비"], r["UA_부하상관"])
    out = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.OUTPUT_DIR / "pinn_params.csv", index=False, encoding="utf-8-sig")
    pd.concat({u: s for u, s in series.items()}).to_frame("ua").to_parquet(
        config.OUTPUT_DIR / "pinn_ua.parquet")
    pd.set_option("display.width", 240)
    print("\n=== 관문1·2: 추정 모수와 물리적 타당성 ===")
    print(out[["히터", "창수", "fire", "C_eff_MJK", "C_eff_배", "Q_loss_kW", "Q_loss_%",
               "UA_중앙_kWK", "설계UA", "설계비", "m설계비", "UA_부하상관", "유효비율"]]
          .round(3).to_string(index=False))
    print("\n  관문1 기준: fire≤1, C_eff배≥1, Q_loss≥0, 설계비 0.2~2.0")
    print("  관문2 기준: |UA_부하상관| < 0.3  (아니면 여전히 유량을 보고 있는 것)")


if __name__ == "__main__":
    main()
