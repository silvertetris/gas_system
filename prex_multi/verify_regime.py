"""1단계 검증 — 운전상태 판정(regime)과 바이패스 분율 β 가 맞는지 (docs §15).

**판정은 ε 하나로 했다.** 그러므로 검증은 ε 이 아닌 **다른 신호**로 해야 의미가 있다.
네 가지를 본다:

  T1 저유량 판정이 맞나 : 그 구간에서 출구온도가 **수조온도**에 붙어 있어야 한다 (ε 아닌 절대온도차).
  T2 차단 판정이 맞나   : 그 구간에서 출구온도가 **입구온도**에 붙어 있어야 한다.
  T3 계열 정합성        : 한 계열의 히터가 **둘 다 차단**이면 데울 경로가 없으니 헤더온도 ≈ 입구온도여야 한다.
                          (regime 은 히터별 ε 로만 만들었고 헤더온도는 거기 안 들어갔다 → 독립 검증)
                          ⚠ `lowflow` 를 여기 넣으면 안 된다 — 저유량은 유량 0 이 아니라 헤더를 올린다.
  T4 β 의 기계적 확인   : 열수지 β 와 밸브개도가 계열별로 단조 대응하는가 (구간별 중앙값).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load() -> pd.DataFrame:
    return pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()


def t1_t2(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u in config.UNITS:
        reg = g[f"regime_{u}"]
        t_in, t_out, t_bath = g[config.TRAIN_INLET[config.TRAIN[u]]], g[f"TI33{u}"], g[f"TI-D2{u}"]
        for r in ["flow", "lowflow", "isolated"]:
            k = reg == r
            if k.sum() < 100:
                continue
            rows.append({
                "히터": u, "상태": r, "n": int(k.sum()),
                "|T_out−T_bath| 중앙": round(float((t_out - t_bath)[k].abs().median()), 2),
                "|T_out−T_in| 중앙": round(float((t_out - t_in)[k].abs().median()), 2),
                "수조−입구 중앙": round(float((t_bath - t_in)[k].median()), 2),
            })
    return pd.DataFrame(rows)


def t3(g: pd.DataFrame) -> pd.DataFrame:
    """계열 히터가 둘 다 비통가스일 때 헤더온도가 입구로 붙는가."""
    rows = []
    for tr, units in config.TRAIN_UNITS.items():
        t_in, t_hdr = g[config.TRAIN_INLET[tr]], g[f"t_hdr_{tr}"]
        regs = [g[f"regime_{u}"] for u in units]
        both_iso = (regs[0] == "isolated") & (regs[1] == "isolated")
        both_low = (regs[0] == "lowflow") & (regs[1] == "lowflow")
        any_flow = (regs[0] == "flow") | (regs[1] == "flow")
        base = t_hdr.notna() & t_in.notna()
        for lab, k in [("둘 다 차단", both_iso & base), ("둘 다 저유량", both_low & base),
                       ("한쪽 이상 통가스", any_flow & base)]:
            if k.sum() < 100:
                continue
            d = (t_hdr - t_in)[k]
            rows.append({"계열": tr, "조건": lab, "n": int(k.sum()),
                         "헤더−입구 중앙[℃]": round(float(d.median()), 2),
                         "p90": round(float(d.quantile(0.9)), 2),
                         "β 중앙": round(float(g[f"beta_{tr}"][k].median()), 3)})
    return pd.DataFrame(rows)


def t4(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    edges = [0, 10, 20, 30, 40, 50, 70, 101]
    for tr in config.TRAIN_UNITS:
        b, v = g[f"beta_{tr}"], g[f"valve_{tr}"]
        k = b.notna() & v.notna()
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = k & (v >= lo) & (v < hi)
            if m.sum() < 500:
                continue
            rows.append({"계열": tr, "밸브개도[%]": f"{lo}~{hi}", "n": int(m.sum()),
                         "β 중앙": round(float(b[m].median()), 3),
                         "β p25": round(float(b[m].quantile(.25)), 3),
                         "β p75": round(float(b[m].quantile(.75)), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    g = load()
    log.info("대상 %s  %s ~ %s", g.shape, g.index.min(), g.index.max())
    pd.set_option("display.width", 220)
    print("\n=== T1·T2  상태별 절대 온도차 (판정에 안 쓴 신호) ===")
    print("   저유량이면 |T_out−T_bath| 이 작아야, 차단이면 |T_out−T_in| 이 작아야 한다")
    print(t1_t2(g).to_string(index=False))
    print("\n=== T3  계열 정합성 (헤더온도는 regime 판정에 안 들어갔다) ===")
    print(t3(g).to_string(index=False))
    print("\n=== T4  β vs 밸브개도 단조성 ===")
    print(t4(g).to_string(index=False))


if __name__ == "__main__":
    main()
