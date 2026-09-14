"""구조 검증 — 어떤 관측량이든 정비 회복을 **대조군 대비** 유의하게 보이는가.

## 왜 이 검정인가

지금까지 여러 지표가 "정비에서 회복하는 것처럼" 보였다가 무너졌다. 마지막 OFF 경로 `U·A` 는
정비 4건 전부 회복(+37.7%)했지만 **아무 시점이나 잡아도 35.8% 움직였다.** 대조군 없이는
회복처럼 보이는 것과 잡음을 구분할 수 없다.

그래서 이번엔 **후보 지표를 전부 같은 틀에 넣고 대조군과 함께 돌린다.** 통과하는 게 하나도 없으면
"이 계측 구성으로는 물리 기반 열화 탐지가 불가능하다"가 확정되고, ML 로 넘어갈 근거가 된다.

## 설계

  · 대상 구간: `regime == "flow"` (열교환이 실제로 일어나는 구간, docs/03)
  · 비교: 정비 시점 기준 **전 365일 vs 후 365일** 중앙값 변화율
    (±180일이면 전/후가 서로 다른 계절이라 계절 confound 가 들어간다. 365일이면 사계절이 균형)
  · **대조군**: 정비에서 200일 이상 떨어진 무작위 시점 200개에 같은 계산
  · 판정: 정비 변화의 절대값이 대조군 절대값 분포에서 **상위 몇 %** 인가
    (단측 순위검정. 0.05 미만이어야 의미 있다)

## 후보 지표

| 기호 | 정의 | 열화 시 기대 |
|---|---|---|
| `eps` | (T_out−T_in)/(T_bath−T_in) | ↓ |
| `ntu` | −ln(1−ε) = U·A/(m·c_p) | ↓ |
| `beta` | 바이패스 분율 | ↓ (제어가 바이패스를 조임) |
| `drive` | T_bath − T_in | ↑ (같은 출력 내려면 수조를 더 데워야) |
| `dT` | T_out − T_in | ↓ |
| `duty` | 버너 가동률 | ↑ (같은 열량 내려면 더 태워야) |

⚠ 다중검정: 지표 6 × 히터 4 = 24 검정이다. 우연히 하나가 0.05 를 밑돌 확률이 70% 가 넘는다.
   그래서 **여러 히터·여러 정비에서 같은 방향으로** 나오는지를 함께 본다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MAINTENANCE = pd.to_datetime(["2013-03-28", "2015-03-16", "2017-03-31", "2019-03-21",
                              "2021-04-02", "2022-10-21", "2024-10-29"])
HALF_WIN_DAYS = 365        # 사계절 균형
MIN_SAMPLES = 2000         # 창당 최소 1분 표본
N_NULL = 200
NULL_GUARD_DAYS = 200      # 정비에서 이만큼 떨어진 시점만 대조군으로


def indicators(g: pd.DataFrame, u: str) -> pd.DataFrame:
    """히터 u 의 후보 지표들. `flow` 구간만."""
    tr = config.TRAIN[u]
    ok = g[f"regime_{u}"] == "flow"
    return pd.DataFrame({
        "eps": g[f"eps_{u}"], "ntu": g[f"ntu_{u}"],
        "beta": g[f"beta_{tr}"], "drive": g[f"drive_{u}"],
        "dT": g[f"dT_{u}"], "duty": g[f"burner_{u}"],
    })[ok]


# 0/1 계열은 중앙값이 0 이라 비율 변화가 정의되지 않는다 → 평균을 쓴다.
# ⚠ 처음엔 이 예외를 안 둬서 `duty` 가 **조용히 검정에서 빠졌다**(24개 중 20개만 돌아감).
BINARY_COLS = {"duty"}


def _change(s: pd.Series, t: pd.Timestamp, col: str = "") -> float:
    w = pd.Timedelta(days=HALF_WIN_DAYS)
    b, a = s[(s.index >= t - w) & (s.index < t)], s[(s.index > t) & (s.index <= t + w)]
    if len(b) < MIN_SAMPLES or len(a) < MIN_SAMPLES:
        return np.nan
    if col in BINARY_COLS:
        mb = b.mean()
        return 100.0 * (a.mean() / mb - 1.0) if mb > 1e-9 else np.nan
    mb = b.median()
    if not np.isfinite(mb) or abs(mb) < 1e-9:
        return np.nan
    return 100.0 * (a.median() / mb - 1.0)


def run() -> pd.DataFrame:
    g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()
    rng = np.random.default_rng(0)
    rows = []
    for u in config.UNITS:
        D = indicators(g, u)
        lo = D.index.min() + pd.Timedelta(days=HALF_WIN_DAYS)
        hi = D.index.max() - pd.Timedelta(days=HALF_WIN_DAYS)
        nulls = []
        while len(nulls) < N_NULL:
            t = lo + pd.Timedelta(days=int(rng.integers(0, max((hi - lo).days, 1))))
            if all(abs((t - m).days) >= NULL_GUARD_DAYS for m in MAINTENANCE):
                nulls.append(t)
        for col in D.columns:
            s = D[col].dropna()
            if len(s) < MIN_SAMPLES * 4:
                continue
            mnt = np.array([_change(s, m, col) for m in MAINTENANCE], dtype=float)
            mnt = mnt[np.isfinite(mnt)]
            nul = np.array([_change(s, t, col) for t in nulls], dtype=float)
            nul = nul[np.isfinite(nul)]
            if len(mnt) < 3 or len(nul) < 30:
                continue
            med = float(np.median(mnt))
            # 단측 순위: 정비 변화의 |중앙| 이 대조군 |변화| 분포에서 상위 몇 % 인가
            p = float(np.mean(np.abs(nul) >= abs(med)))
            rows.append({"히터": u, "지표": col, "정비n": len(mnt),
                         "정비변화중앙%": round(med, 1),
                         "회복건수": int(np.sum(mnt > 0)),
                         "대조군n": len(nul),
                         "대조군|변화|중앙%": round(float(np.median(np.abs(nul))), 1),
                         "p(단측)": round(p, 3),
                         "유의": "✅" if p < 0.05 else ""})
            log.info("[%s] %-6s 정비 %+7.1f%% (회복 %d/%d) | 대조군 |Δ| %5.1f%% | p=%.3f%s",
                     u, col, med, int(np.sum(mnt > 0)), len(mnt),
                     float(np.median(np.abs(nul))), p, "  ✅" if p < 0.05 else "")
    return pd.DataFrame(rows)


def main() -> None:
    out = run()
    out.to_csv(config.OUTPUT_DIR / "verify_signal.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 정비 회복 검정 (대조군 대비) ===")
    print(out.sort_values("p(단측)").to_string(index=False))
    n_sig = int((out["p(단측)"] < 0.05).sum())
    print(f"\n검정 {len(out)}개 중 p<0.05 인 것 {n_sig}개 "
          f"(우연 기대값 {0.05*len(out):.1f}개)")


if __name__ == "__main__":
    main()
