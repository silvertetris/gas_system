"""열매체액 누출 추적 — Degradation(트랙2) + RUL(트랙3).

## 왜 이 트랙인가

다른 트랙은 검증에서 죽었다(docs/07):
  · 트랙1 Hard fault — SCADA 에 고장 개념이 없다. 알람은 공정 이탈 알림이거나
    자동복구 트랜지언트다(화염검출의 89%가 6시간 내 재점화, 수온 알람의 94%가 정지 중).
  · 트랙2 열전달 열화 — `U·A` 는 유량 미계측으로 산출 불가, 대리지표 24개 전부 검정 실패(최소 p=0.465).

**누출은 다르다.** 열화 메커니즘이 **직접 계측**되고(수위), 고장 임계가 **설계로 정해져 있다**(L-L 정지).
유량도, 유효 화력률도, 에너지 수지도 필요 없다.

    누출 → 수위 하강 → 보충(톱니) → **하강률 = 누출률** → L-L 도달시간 = RUL

도메인 근거: 2025 고장경향성 보고서에서 가스히터 기계고장 41건 중 **열매체액 충진/보충이 7건(17%)**,
경기지사 사례는 **Tube Bundle 부식으로 Hole 발생 → 열매체액 외부 누수**다(docs/05).

⚠ **P 는 수위계가 없다.** A/B/O 만 가능하다(docs/01).

## ⚠️ 온도 보정이 1순위다

수위는 열매체액 **열팽창**에 따라 변한다. 수조가 20℃일 때와 60℃일 때 수위가 다르다.
보정하지 않으면 **계절 변동을 누출로 오독**한다. 그래서 먼저 팽창계수를 실측한다:
짧은 창(누출이 무시할 만한 시간) 안에서 수위 변화를 수조온도 변화에 회귀한다.

## 단위

관측 범위가 0~100 이고 설계 임계는 H 950 / L 150 / L-L −150 mm 다 → **퍼센트 스케일**로 본다.
스팬 1,100 mm 가정 시 **1% ≈ 11 mm**, L-L(−150mm) ≈ **0%**, L(150mm) ≈ **27%**.
⚠ 이 환산은 제작도서로 확인되지 않았다. 절대 RUL 을 쓸 때는 반드시 검증할 것.
상대 추세(누출률의 시간 변화)는 환산과 무관하다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

UNITS = ["A", "B", "O"]                 # P 는 수위계 미설치
LEVEL_MIN, LEVEL_MAX = 1.0, 99.0        # 계기 끝단(0/100)은 dropout 으로 본다
REFILL_JUMP = 2.0                       # 1시간에 이만큼 오르면 보충 [%]
REFILL_WINDOW_MIN = 60
MIN_CYCLE_HOURS = 72                    # 하강률을 잴 최소 사이클 길이
MIN_CYCLE_ROWS = 2000
EXPANSION_WIN_H = 6                     # 팽창계수 추정용 창 (이 안에서는 누출 무시)
LL_THRESHOLD_PCT = 0.0                  # L-L 정지 인터록 (−150 mm ≈ 0%)


def thermal_coefficient(g: pd.DataFrame, u: str) -> tuple[float, float, int]:
    """열팽창계수 k = dLevel/dT_bath [%/℃]. 짧은 창 안의 변화량끼리 회귀."""
    lv, tb = g[f"LI-D1{u}"], g[f"TI-D2{u}"]
    ok = lv.between(LEVEL_MIN, LEVEL_MAX) & tb.notna()
    key = g.index.floor(f"{EXPANSION_WIN_H}h")
    d = pd.DataFrame({"k": key[ok], "lv": lv[ok], "tb": tb[ok]})
    agg = d.groupby("k").agg(dl=("lv", lambda s: s.iloc[-1] - s.iloc[0]),
                             dt=("tb", lambda s: s.iloc[-1] - s.iloc[0]),
                             n=("lv", "size"))
    agg = agg[(agg["n"] > EXPANSION_WIN_H * 60 * 0.8) & (agg["dt"].abs() > 2.0)
              & (agg["dl"].abs() < REFILL_JUMP)]     # 보충 창은 제외
    if len(agg) < 200:
        return 0.0, np.nan, len(agg)
    x, y = agg["dt"].to_numpy(), agg["dl"].to_numpy()
    k = float(np.sum(x * y) / np.sum(x * x))          # 절편 없는 회귀 (ΔT=0 이면 ΔL=0)
    r = y - k * x
    r2 = float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2))
    return k, r2, len(agg)


def cycles(g: pd.DataFrame, u: str, k_exp: float) -> pd.DataFrame:
    """보충 사이 구간마다 하강률(누출률)을 잰다."""
    lv, tb = g[f"LI-D1{u}"], g[f"TI-D2{u}"]
    ok = lv.between(LEVEL_MIN, LEVEL_MAX) & tb.notna()
    # 온도 보정: 기준온도 대비 팽창분을 뺀다
    t_ref = float(tb[ok].median())
    corr = (lv - k_exp * (tb - t_ref))[ok]

    jump = corr.diff(REFILL_WINDOW_MIN)
    refill = (jump > REFILL_JUMP).fillna(False)
    cyc = (refill & ~refill.shift(fill_value=False)).cumsum()

    rows = []
    for cid, s in corr.groupby(cyc):
        s = s[~refill.reindex(s.index, fill_value=False)]
        if len(s) < MIN_CYCLE_ROWS:
            continue
        hours = (s.index[-1] - s.index[0]).total_seconds() / 3600
        if hours < MIN_CYCLE_HOURS:
            continue
        t = (s.index - s.index[0]).total_seconds().to_numpy() / 86400.0   # 일
        y = s.to_numpy()
        tc, yc = t - t.mean(), y - y.mean()
        slope = float(np.sum(tc * yc) / np.sum(tc * tc))                  # %/day
        resid = yc - slope * tc
        r2 = float(1 - np.sum(resid ** 2) / np.sum(yc ** 2)) if np.sum(yc ** 2) > 0 else np.nan
        rows.append({"start": s.index[0], "end": s.index[-1], "hours": hours,
                     "level0": float(y[0]), "level1": float(y[-1]),
                     "leak_pct_day": -slope, "r2": r2, "n": len(s)})
    return pd.DataFrame(rows)


def main() -> None:
    cols = ["Time", "segment_id"] + [f"LI-D1{u}" for u in UNITS] + [f"TI-D2{u}" for u in UNITS]
    g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet",
                        columns=cols).set_index("Time").sort_index()
    out = {}
    for u in UNITS:
        k, r2, n = thermal_coefficient(g, u)
        log.info("[%s] 열팽창계수 k = %+.4f %%/℃  (R²=%.3f, 창 %d개)", u, k, r2, n)
        C = cycles(g, u, k)
        out[u] = C
        if C.empty:
            log.info("[%s] 사이클 없음", u)
            continue
        pos = C[C["leak_pct_day"] > 0]
        log.info("[%s] 보충 사이클 %d개 | 길이 중앙 %.0fh | 누출률 중앙 %+.4f %%/day "
                 "(하강 사이클 %d개, 그 중앙 %+.4f)",
                 u, len(C), C["hours"].median(), C["leak_pct_day"].median(),
                 len(pos), pos["leak_pct_day"].median() if len(pos) else np.nan)
        C.assign(unit=u).to_csv(config.OUTPUT_DIR / f"leak_cycles_{u}.csv",
                                index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 연도별 누출률 중앙 [%/day] — 양수가 하강(누출) ===")
    print("%4s" % "", end="")
    yrs = list(range(2012, 2027))
    for y in yrs:
        print("%8d" % y, end="")
    print()
    for u in UNITS:
        C = out[u]
        if C.empty:
            continue
        m = C.assign(yr=pd.to_datetime(C["start"]).dt.year).groupby("yr")["leak_pct_day"].median()
        print("%4s" % u, end="")
        for y in yrs:
            v = m.get(y, np.nan)
            print("%8.3f" % v if np.isfinite(v) else "%8s" % "-", end="")
        print()


if __name__ == "__main__":
    main()
