"""[진단] 정비 주기별 열화 추세 — KF-PINN 의 핵심 전제를 검증한다.

**검증할 가설**: 히터는 정비 사이에 서서히 열화(`U·A`↓)하고, 정비가 그것을 되돌린다.
  즉 `U·A` 궤적이 **톱니(sawtooth)** 모양이어야 한다.

**왜 이 검증이 필요한가**: docs §8-3 에서 "열화 추세 없음"으로 결론냈는데, 그건 전 구간을
한 줄로 회귀한 결과였다. 그 사이에 정기점검이 7번 있었고 정비는 `U·A` 를 계단식으로 회복시키므로
(§9-3b 실측: 2021 ε 0.182→0.236, 2024 0.273→0.640), 열화와 회복이 **상쇄돼 평평해 보인 것**이다.
정비로 끊어서 봐야 한다.

**지표**: ε = (T_out−T_in)/(T_bath−T_in). NTU = −ln(1−ε) = U·A/(m_gas·c_p).
  신뢰 태그 3개(TI21Z/TI33P/TI-D2P)만 쓰므로 P&ID 로 흔들린 재료의 영향을 받지 않는다(§7).

⚠ **한계 — 부하 보정이 안 돼 있다**: NTU 는 `U·A` 와 `m_gas` 의 비다. 부하가 늘면 NTU 가
  떨어지므로, 여기서 보이는 하락이 열화인지 부하 증가인지 **이 단계에서는 못 가른다**.
  그게 2단계(U·A = UA_ref·(m/m_design)^0.8) 의 몫이다. 여기서는 **톱니 패턴의 존재 여부**만 본다.

실행: .venv/bin/python -m prex.degradation
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

MIN_WORK_DT = 5.0        # 승온폭이 이보다 커야 '히터가 일하는 중' (docs §8-4)
MIN_DAYS_IN_CYCLE = 30   # 이보다 짧은 주기는 추세를 못 본다
MIN_SAMPLES_PER_DAY = 60  # 하루에 최소 이만큼 유효 표본이 있어야 그 날을 쓴다


def load_trend() -> pd.DataFrame:
    p = config.OUTPUT_DIR / "trend_htr31p.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — `python -m prex.pipeline` 먼저 실행")
    return pd.read_parquet(p).set_index("Time").sort_index()


def daily_eps(df: pd.DataFrame) -> pd.DataFrame:
    """'히터가 일하는 중'인 시각만 골라 일별 ε 중앙값을 만든다.

    노는 구간을 섞으면 ε 가 가동률을 반영해 버린다(§8-3 에서 저지른 실수).
    """
    work = ((df["TI33P"] - df[config.HTR31P_INLET_TAG]) > MIN_WORK_DT) & df["htx_eps"].notna()
    d = df.loc[work, ["htx_eps", "htx_ntu", "maint_cycle", "days_since_maint"]]
    g = d.resample("1D").agg(eps=("htx_eps", "median"), ntu=("htx_ntu", "median"),
                             n=("htx_eps", "size"),
                             cycle=("maint_cycle", "median"),
                             days=("days_since_maint", "median"))
    g = g[g["n"] >= MIN_SAMPLES_PER_DAY].dropna(subset=["eps", "cycle"])
    logger.info("일별 ε: %d일 (가동 중 & 유효)", len(g))
    return g


def fit_cycle_trends(g: pd.DataFrame) -> pd.DataFrame:
    """주기별로 ε ~ 경과일 선형회귀. 기울기가 음수면 열화 방향."""
    rows = []
    for cyc, sub in g.groupby("cycle"):
        if len(sub) < MIN_DAYS_IN_CYCLE:
            continue
        x, y = sub["days"].to_numpy(), sub["eps"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append({
            "cycle": int(cyc), "시작": sub.index.min().date(), "종료": sub.index.max().date(),
            "일수": len(sub), "기간일": round(float(x.max() - x.min())),
            "eps_시작": round(float(intercept), 4),
            "eps_종료": round(float(slope * x.max() + intercept), 4),
            "기울기_per_100d": round(float(slope * 100), 5),
            "r2": round(float(r2), 3),
            "eps_중앙": round(float(np.median(y)), 4),
        })
    return pd.DataFrame(rows)


def maintenance_jumps(g: pd.DataFrame, window_days: int = 60) -> pd.DataFrame:
    """정비 직전/직후 ε 비교 — 정비가 실제로 회복시키는가."""
    rows = []
    cycles = sorted(g["cycle"].unique())
    for a, b in zip(cycles[:-1], cycles[1:]):
        before = g[(g["cycle"] == a)].tail(window_days)["eps"]
        after = g[(g["cycle"] == b)].head(window_days)["eps"]
        if len(before) < 10 or len(after) < 10:
            continue
        rows.append({
            "정비": f"cycle {int(a)}→{int(b)}",
            "시점": g[g["cycle"] == b].index.min().date(),
            "직전_eps": round(float(before.median()), 4),
            "직후_eps": round(float(after.median()), 4),
            "변화": round(float(after.median() - before.median()), 4),
            "변화율_%": round(100 * float(after.median() - before.median()) / float(before.median()), 1),
        })
    return pd.DataFrame(rows)


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_trend()
    g = daily_eps(df)
    trends = fit_cycle_trends(g)
    jumps = maintenance_jumps(g)
    return g, trends, jumps


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 170)
    g, trends, jumps = run()

    print("\n=== 정비 주기별 ε 추세 (기울기 음수 = 열화 방향) ===")
    print(trends.to_string(index=False))
    n_neg = int((trends["기울기_per_100d"] < 0).sum())
    print(f"\n  하락 주기 {n_neg}/{len(trends)}개  "
          f"| 평균 기울기 {trends['기울기_per_100d'].mean():+.5f} /100일")

    print("\n=== 정비 전후 ε 변화 (양수 = 정비가 회복시킴) ===")
    print(jumps.to_string(index=False))
    n_up = int((jumps["변화"] > 0).sum())
    print(f"\n  상승 {n_up}/{len(jumps)}회  | 평균 {jumps['변화_%' if '변화_%' in jumps else '변화율_%'].mean():+.1f}%")

    print("\n=== 톱니 패턴 판정 ===")
    ok_down = n_neg >= len(trends) * 0.6
    ok_up = n_up >= len(jumps) * 0.6
    print(f"  주기 내 하락 우세: {'✅' if ok_down else '❌'} ({n_neg}/{len(trends)})")
    print(f"  정비 후 상승 우세: {'✅' if ok_up else '❌'} ({n_up}/{len(jumps)})")
    print(f"  → {'톱니 패턴 확인 — KF-PINN 전제 성립' if ok_down and ok_up else '톱니 미확인 — 재검토 필요'}")

    out = config.OUTPUT_DIR / "degradation"
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "daily_eps.csv")
    trends.to_csv(out / "cycle_trends.csv", index=False)
    jumps.to_csv(out / "maintenance_jumps.csv", index=False)
    logger.info("저장: %s", out)


if __name__ == "__main__":
    main()
