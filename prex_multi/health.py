"""열화지표 `U·A(t)` — **버너 OFF 구간의 냉각률**만 사용한다 (2026-09-12 재설계).

## 왜 OFF 만 쓰나

수조 열수지에서 버너가 꺼져 있으면 구동항이 사라진다:

    C_eff · dT_bath/dt = −Q_gas − Q_loss        →    Q_gas = C_eff·|dT/dt| − Q_loss
    U·A = m·c_p·NTU,   m·c_p = Q_gas/Δ          →    **U·A = Q_gas · NTU / Δ**

정리하면 `rate_loss ≡ Q_loss/C_eff`(무부하 냉각률, 실측)를 써서

    **U·A = C_eff · (|dT_bath/dt| − rate_loss) · NTU / Δ**

괄호 안이 **순수 실측 냉각률의 차이**다. 이 식에는 다음이 **들어가지 않는다**:
  · 설계 열량 `Q_b`      — A/B/O 는 T3 프록시라 못 믿는다
  · 유효 화력률 `fire`   — 끝내 식별되지 않았다(호기별 0.35~0.8 로 흩어짐)
  · 버너 가동률 `s̄` 의 **크기** — 오직 "OFF 다"라는 상태만 쓴다
  · 유량 `m_tot`·`f_i`·합류점 식 (H5)·부하보정 (H4)

`C_eff` 는 `fire` 만큼 불확정이지만 **시불변 상수**라 시간 추세·비율에서 소거된다.

## OFF 를 믿어도 되는 근거 (실측 검증)

DI 로그는 상태변화 시점만 기록하므로 전처리에서 직전 상태로 채운다(§02). 그 채움이
긴 구간을 덮을 때 믿을 수 있는지 확인했다 — 창별 수조 승온율로 교차검증:

| 창 종류 | A | B | O | P |
|---|---|---|---|---|
| `s̄=0` & 블록 >6h — **실제 냉각 비율** | **99.8%** | **99.6%** | **99.7%** | **97.5%** |
| `s̄=1` & 블록 >6h — 승온 비율 | 21.4% | 23.2% | 42.0% | 42.9% |
| `0.3<s̄<0.9` (정상 사이클) — 승온 비율 | **79.2%** | 31.4% | 52.2% | 48.1% |

**긴 OFF 는 진짜 OFF 다** — 히터가 정지해 있으니 상태가 안 바뀌는 것이 당연하고,
전방채움이 맞는 값을 준다. 반대로 **긴 ON 은 오염돼 있다**: A 호기에서 정상 사이클은
+10.23 ℃/h 로 79.2% 가 승온하는데 얼어붙은 ON 은 −0.38 ℃/h 로 21.4% 만 승온한다.
버너가 켜져 있다면서 식고 있다.

→ **신뢰되는 신호(OFF)만 쓰고 오염된 신호(ON 지속시간)는 안 쓴다.** 데이터를 고치거나
전방채움을 제한할 필요가 없다 — 애초에 안 쓰면 된다.

## 기각: 버너 ON 열량 기반 (구 버전)

    U·A = (fire·s̄·Q_b − Q_loss − C_eff·dT/dt)·NTU/Δ
`s̄` 의 크기를 쓰므로 얼어붙은 ON 이 그대로 들어간다. 실제로 P 호기 고부하 창의 27.8% 가
`s̄=1.0` 이었고 그중 85.8% 가 6시간 초과 블록이었다. 그 창들을 빼면 `U·A` 가
13.9 → 11.7 kW/K 로 16% 내려갔다 — 조용한 상향 편향이다.

## 남는 가정
  1. `C_eff`·`Q_loss` 가 시간에 대해 거의 상수
  2. 열교환 해석이 유효한 `regime == "flow"` 구간만 사용
  3. 냉각률이 무부하 냉각률보다 확실히 클 것(아니면 가스가 열을 빼가지 않는 것)

## 검증 기준
정기점검(튜브 세척)에서 `U·A` 가 계단식으로 회복해야 한다. 안 보이면 이 지표는 열화를 못 본다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import calorimetry, config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

WINDOW_MIN = 60
MIN_COVERAGE = 0.9         # 창 내 수조온도 유효 비율
MIN_DELTA_C = 5.0          # 가스 승온폭 하한 — 작으면 U·A 가 발산
MIN_SAMPLES = 30           # 기울기 회귀 최소 표본
COOL_MARGIN = 1.5          # |냉각률| 이 무부하 냉각률의 몇 배를 넘어야 하나
STALE_HOURS = 6            # 이 시간 초과로 상태가 안 바뀌면 '채움 지속' 플래그 (분석용)

# 정기점검 (docs/05). 튜브 세척으로 U·A 가 회복돼야 하는 시점.
MAINTENANCE = ["2013-03-28", "2015-03-16", "2017-03-31", "2019-03-21",
               "2021-04-02", "2022-10-21", "2024-10-29"]


def _window_slope(x: pd.Series, key: np.ndarray) -> pd.DataFrame:
    """창별 최소제곱 기울기 [K/s]와 표본수. `TI-D2x` 분해능 0.061℃ 라 단순 차분은 계단이 된다.

    Σ(t−t̄)(y−ȳ)/Σ(t−t̄)² 를 groupby 합계만으로 계산한다(apply 는 13만 그룹에서 너무 느리다).
    """
    f = pd.DataFrame({"g": key, "y": x.to_numpy()}).dropna()
    f["t"] = f.groupby("g").cumcount().to_numpy() * 60.0
    g = f.groupby("g")
    n = g.size()
    st, sy = g["t"].sum(), g["y"].sum()
    stt, sty = (f["t"] ** 2).groupby(f["g"]).sum(), (f["t"] * f["y"]).groupby(f["g"]).sum()
    num = sty - st * sy / n
    den = stt - st ** 2 / n
    return pd.DataFrame({"rate": (num / den.replace(0.0, np.nan)), "n": n})


def windows(g: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """버너 OFF · 통가스 창에서 `U·A` 를 계산한다."""
    key = g.index.floor(f"{WINDOW_MIN}min")
    frames = []
    for u in config.UNITS:
        t_in = g[config.TRAIN_INLET[config.TRAIN[u]]]
        burner = g[f"burner_{u}"]
        # 상태가 얼마나 오래 안 바뀌었나 (데이터 수정 아님 — 메타데이터)
        blk = (burner != burner.shift()).cumsum()
        hold = burner.groupby(blk).transform("size")

        sl = _window_slope(g[f"TI-D2{u}"], key)
        W = pd.DataFrame({
            "s": burner.groupby(key).mean(),
            "d": (g[f"TI33{u}"] - t_in).groupby(key).median(),
            "ntu": g[f"ntu_{u}"].groupby(key).median(),
            "eps": g[f"eps_{u}"].groupby(key).median(),
            "t_bath": g[f"TI-D2{u}"].groupby(key).median(),
            "t_in": t_in.groupby(key).median(),
            "cov": g[f"TI-D2{u}"].notna().groupby(key).mean(),
            "hold_min": pd.Series(hold.to_numpy(), index=g.index).groupby(key).max(),
            "reg": g[f"regime_{u}"].groupby(key).agg(
                lambda s: s.iloc[0] if s.notna().all() and s.nunique() == 1 else None),
        }).join(sl)

        c_eff = float(cal.loc[u, "C_eff_MJK"]) * 1e6
        rate_loss = float(cal.loc[u, "냉각율_Ch"]) / 3600.0          # K/s, 양수
        cool = -W["rate"]                                            # 냉각이면 양수
        W["q_gas"] = c_eff * (cool - rate_loss)
        W["ua"] = W["q_gas"] * W["ntu"] / W["d"]
        W["unit"] = u
        W["stale"] = W["hold_min"] > STALE_HOURS * 60                # 분석용 플래그

        ok = ((W["s"] == 0)                                          # ★ 버너 OFF 만
              & (W["reg"] == "flow") & (W["d"] > MIN_DELTA_C)
              & (W["cov"] > MIN_COVERAGE) & (W["n"] >= MIN_SAMPLES)
              & (cool > COOL_MARGIN * rate_loss)                     # 가스가 확실히 열을 빼감
              & W["ntu"].notna() & np.isfinite(W["ua"]))
        frames.append(W[ok])
        log.info("[%s] OFF·통가스 창 %d개 | U·A 중앙 %.1f kW/K | 채움지속 플래그 %.1f%%",
                 u, int(ok.sum()), float(W.loc[ok, "ua"].median()) / 1e3,
                 100 * float(W.loc[ok, "stale"].mean()))
    return pd.concat(frames)


def main() -> None:
    g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()
    cal = calorimetry.measure(g).set_index("히터")
    W = windows(g, cal)
    W.to_parquet(config.OUTPUT_DIR / "health_windows.parquet")

    pd.set_option("display.width", 220)
    print("\n=== U·A [kW/K] — 설계값과 비교 ===")
    rows = []
    for u in config.UNITS:
        s = W[W["unit"] == u]
        des = (config.P_UA_W_PER_K if u == "P" else config.scaled_ua_w_per_k(u)) / 1e3
        rows.append({"히터": u, "창수": len(s),
                     "U·A중앙": round(s["ua"].median() / 1e3, 2),
                     "p25": round(s["ua"].quantile(.25) / 1e3, 2),
                     "p75": round(s["ua"].quantile(.75) / 1e3, 2),
                     "설계": round(des, 1), "설계비": round(s["ua"].median() / 1e3 / des, 2),
                     "등급": "T1" if u == "P" else "T3(프록시)"})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 민감도: 채움지속(>6h) 창 포함 여부 ===")
    for u in config.UNITS:
        s = W[W["unit"] == u]
        a, b = s[~s["stale"]], s[s["stale"]]
        if len(a) < 50 or len(b) < 50:
            print(f"  {u}: 표본 부족 ({len(a)}/{len(b)})")
            continue
        print(f"  {u}: 전체 {s['ua'].median()/1e3:6.2f} | 비플래그 {a['ua'].median()/1e3:6.2f}"
              f" (n={len(a):5d}) | 플래그 {b['ua'].median()/1e3:6.2f} (n={len(b):5d})"
              f" | 차이 {100*(b['ua'].median()/a['ua'].median()-1):+.1f}%")


if __name__ == "__main__":
    main()
