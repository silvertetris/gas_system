"""무부하 열량계 — `C_eff` 와 `Q_loss` 를 데이터만으로 측정한다.

**왜 필요한가**: 버너 가동률 유량 역산(`backsolve.py`)은 `m_i = (s̄·Q_b − Q_loss)/(c_p·Δ_i)` 인데
`Q_loss` 를 "차단 구간의 평균 버너 가동률 × Q_b" 로 잡았던 것이 **틀렸다.**
실측해 보니 **차단 구간의 86~92% 는 버너 가동률이 정확히 0** 이다 — 히터가 완전 정지 상태라
그 가동률은 손실을 상쇄하는 값이 아니다. 결과적으로 `Q_loss ≈ 0` 을 쓴 셈이고 `m_i` 가 과대평가됐다
(O 호기가 설계 유량의 1.25배로 나온 원인).

**어떻게 고치나** — 차단 구간은 가스 흡수열이 0 이므로 순수한 열량계가 된다. 두 가지를 분리해 잰다:

  (1) **승온 에피소드** (버너 ON, 가스 0):  `C_eff·ΔT = fire·Q_b·t_on`
      → `fire·C_eff = Q_b·t_on/ΔT`
  (2) **냉각 구간**   (버너 OFF, 가스 0):  `C_eff·|dT/dt| = Q_loss`

### 무엇이 측정되고 무엇이 안 되나 (2026-09-12 재검토)

(1)이 주는 것은 **곱 `M = fire·C_eff` 하나**다. 이 곱은 저가동률(<0.2)에서 안정적으로 측정된다
— A 호기 가동률 0.077/0.121/0.273 에서 35.9 / 38.7 / 37.4 MJ/K, 95% 신뢰구간이 서로 겹치고
가동률과의 Spearman 이 −0.058 이다. **`fire` 가 그 범위에서 상수라는 증거는 된다.**

**그러나 `fire` 의 값 자체는 측정되지 않는다.** `fire = 1` 은 ON/OFF 사이클 버너가 켜지면
정격 화력이라는 **기기 상식에 근거한 가정**이지 이 데이터가 준 값이 아니다.
고가동률(≥0.6) 차단 승온 에피소드가 있으면 검증되겠으나 표본이 없다
(P 의 0.2~0.35 구간 신뢰구간이 [51, 904] MJ/K 일 만큼 넓다).

⚠ **`fire` 를 바꾸면 무엇이 따라 움직이나** (앞서 한 번 잘못 적었다):
```
C_eff  = M / fire
Q_loss = C_eff·rate = M·rate / fire        ← fire 에 **반비례**한다
m_i    = (s̄·fire·Q_b − M·rate/fire) / (c_p·Δ)
```
`Q_loss` 가 `fire` 에 비례한다고 적었던 것은 **틀렸다.** 반비례다.
그리고 `m_i` 는 `fire` 가 줄면 **선형보다 빠르게** 줄어든다(앞 항은 줄고 빼는 항은 커지므로).

**결과적으로 `m_i` 의 절대 크기는 `fire` 만큼 불확정이다.** 다만 `fire` 가 시간에 대해 상수라면
**`m_i(t)` 의 시간 추세는 영향을 덜 받는다** — 열화지표 `φ` 의 검증 기준이 "정비 시점에서 회복하는가"
라는 **형태 검정**이라 이 점이 중요하다. 절대 수준은 `fire` 불확정성을 명시해 보고할 것.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BURNER_EFF = 0.904        # 제작도서 연소효율
MIN_BATH_C = 15.0
HEAT_MIN_DT_C = 8.0       # 승온 에피소드 최소 상승폭
HEAT_MIN_HOURS = 1.0
HEAT_MIN_ON_HOURS = 0.2
COOL_MIN_LEN = 30         # 냉각 런 최소 길이 [분]
COOL_SETTLE = 10          # OFF 직후 버림 [분]
FIRE_FACTOR = 1.0         # 위 근거. 바꾸려면 C_eff·Q_loss 가 같이 스케일됨


def _isolated_runs(g: pd.DataFrame, u: str) -> tuple[pd.Series, pd.Series]:
    bath = g[f"TI-D2{u}"]
    ok = ((g[f"regime_{u}"] == "isolated") & bath.notna()
          & g[f"burner_{u}"].notna() & (bath > MIN_BATH_C))
    run = ((ok != ok.shift()) | (g["segment_id"] != g["segment_id"].shift())).cumsum()
    return ok, run


def c_eff_from_heating(g: pd.DataFrame, u: str) -> dict:
    """차단 상태 승온 에피소드 적분 → `fire·C_eff`. `fire=1` 로 두면 `C_eff`."""
    ok, run = _isolated_runs(g, u)
    bath, bn = g[f"TI-D2{u}"], g[f"burner_{u}"]
    qb = config.design_duty_w(u) / BURNER_EFF
    rows = []
    for _, s in bath[ok].groupby(run[ok]):
        if len(s) < 60:
            continue
        d_t, hours = float(s.iloc[-1] - s.iloc[0]), len(s) / 60.0
        if d_t < HEAT_MIN_DT_C or hours < HEAT_MIN_HOURS:
            continue
        t_on = float(bn.loc[s.index].sum()) / 60.0
        if t_on < HEAT_MIN_ON_HOURS:
            continue
        rows.append({"dT": d_t, "duty": t_on / hours, "prod": qb * t_on * 3600.0 / d_t})
    R = pd.DataFrame(rows)
    if len(R) < 10:
        return {"c_eff": np.nan, "n": len(R), "duty_med": np.nan, "iqr": np.nan}
    p = R["prod"]
    return {"c_eff": float(p.median()) / FIRE_FACTOR, "n": len(R),
            "duty_med": float(R["duty"].median()),
            "iqr": float(p.quantile(0.75) - p.quantile(0.25))}


def q_loss_from_cooling(g: pd.DataFrame, u: str, c_eff: float) -> dict:
    """차단 + 버너 OFF 냉각 런 → `Q_loss = C_eff·|dT/dt|`. 런 내부 최소제곱 기울기(양자화 대응)."""
    ok, run = _isolated_runs(g, u)
    ok = ok & (g[f"burner_{u}"] == 0)
    run = ((ok != ok.shift()) | (g["segment_id"] != g["segment_id"].shift())).cumsum()
    bath = g[f"TI-D2{u}"]
    f = pd.DataFrame({"run": run[ok], "T": bath[ok].to_numpy(),
                      "dTa": (bath - g[config.TRAIN_INLET[config.TRAIN[u]]])[ok].to_numpy()})
    f = f[f.groupby("run").cumcount() >= COOL_SETTLE]
    if f.empty:
        return {"q_loss_kw": np.nan, "n": 0, "rate": np.nan, "dTa": np.nan}
    f["t"] = f.groupby("run").cumcount().to_numpy() * 60.0
    gr = f.groupby("run")
    n, tb, Tb = gr.size(), gr["t"].mean(), gr["T"].mean()
    stt = gr["t"].apply(lambda s: float(((s - s.mean()) ** 2).sum()))
    stT = f.assign(tT=f["t"] * f["T"]).groupby("run")["tT"].sum() - n * tb * Tb
    R = pd.DataFrame({"L": n, "slope": stT / stt.replace(0.0, np.nan), "dTa": gr["dTa"].median()})
    R = R[(R["L"] >= COOL_MIN_LEN) & R["slope"].notna() & (R["slope"] < 0)]
    if len(R) < 30 or not np.isfinite(c_eff):
        return {"q_loss_kw": np.nan, "n": len(R), "rate": np.nan, "dTa": np.nan}
    rate = float((-R["slope"]).median())                      # K/s
    return {"q_loss_kw": c_eff * rate / 1e3, "n": len(R),
            "rate": rate * 3600.0, "dTa": float(R["dTa"].median())}


def measure(g: pd.DataFrame | None = None) -> pd.DataFrame:
    if g is None:
        g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()
    rows = []
    for u in config.UNITS:
        h = c_eff_from_heating(g, u)
        q = q_loss_from_cooling(g, u, h["c_eff"])
        sens = config.scaled_mw_cw_j_per_k(u)
        duty = config.design_duty_w(u)
        rows.append({
            "히터": u, "승온에피소드": h["n"], "가동률중앙": round(h["duty_med"], 3),
            "C_eff_MJK": round(h["c_eff"] / 1e6, 1),
            "설계현열_MJK": round(sens / 1e6, 1),
            "현열대비": round(h["c_eff"] / sens, 2),
            "냉각런": q["n"], "냉각율_Ch": round(q["rate"], 2), "수조−입구_C": round(q["dTa"], 1),
            "Q_loss_kW": round(q["q_loss_kw"], 1),
            "설계열량대비%": round(100 * q["q_loss_kw"] * 1e3 / duty, 2),
        })
        log.info("[%s] C_eff %.1f MJ/K (현열의 %.2f배, 에피소드 %d) | 냉각 %.2f ℃/h → Q_loss %.1f kW (%.2f%%)",
                 u, h["c_eff"] / 1e6, h["c_eff"] / sens, h["n"],
                 q["rate"], q["q_loss_kw"], 100 * q["q_loss_kw"] * 1e3 / duty)
    out = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.OUTPUT_DIR / "calorimetry.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(measure().to_string(index=False))
