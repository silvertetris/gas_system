"""정제 비교 실험 — 넣기 전 vs 넣은 후(각각) vs 세 가지 모두. 시드 반복·짝비교.

## 비교 구조

| 구분 | 조합 |
|---|---|
| 넣기 전 (기준) | 없음 |
| 넣은 후, 서로 비교 | ＋KF · ＋AE · ＋EKF |
| 세 가지 모두 | ＋KF·AE·EKF |

AE·KF·EKF 는 **정제용**이다 — 출력은 PINN 신경망 입력으로만 들어간다. 셋 중 하나를 고르는
실험이 아니라, **각각 넣었을 때와 모두 넣었을 때를 같은 조건에서 비교**하는 실험이다.

## 왜 시드를 반복하나

분할(학습 ~2019 · 검증 ~2022 · 시험 2022~)은 고정이고 바뀌는 것은 **시드**(신경망 초기값·
몬테카를로 표본·GBM 난수)뿐이다. 단일 시드로는 AUC ±0.02, 리프트1%(상위 약 335행) 차이가
잡음인지 효과인지 가를 수 없다. 그래서

  · 조합별 **평균 ± 표준편차**
  · **같은 시드끼리 짝지은 차이** (조합 − 없음, 그리고 조합끼리) 와 개선된 시드 수

로 비교한다. 짝비교는 시드 간 공통 변동을 상쇄해 차이를 더 날카롭게 본다.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from . import config, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

VARIANTS = [("없음", ()), ("＋KF", ("kf",)), ("＋AE", ("ae",)), ("＋EKF", ("ekf",)),
            ("＋KF·AE·EKF", ("kf", "ae", "ekf"))]
SEEDS = (42, 7, 123, 2024, 31337)
METRICS = ["PINN_전체_AUC", "PINN_전체_리프트1", "PINN_전체_리프트5", "PINN_전체_Brier개선%",
           "PINN_유량관측_AUC", "PINN_유량미관측_AUC", "공급온도예측RMSE",
           "GBM_전체_AUC", "GBM_전체_리프트1", "μ_JT", "UA_배관_kW/K", "N₀"]
# 짝비교에 쓸 핵심 지표와 방향 (+1 = 클수록 좋다)
PAIR_METRICS = {"PINN_전체_AUC": +1, "PINN_전체_리프트1": +1, "PINN_전체_리프트5": +1,
                "PINN_전체_Brier개선%": +1, "공급온도예측RMSE": -1}


def run_all() -> pd.DataFrame:
    """이미 끝난 (시드, 조합, 계열)은 건너뛴다 — 중단 뒤 재개할 수 있게."""
    raw_path = config.OUTPUT_DIR / "refine_seeds_raw.csv"
    done = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    have = set(zip(done["시드"], done["정제"], done["계열"])) if len(done) else set()
    if have:
        log.info("재개: 완료 %d건은 건너뛴다", len(have))
    rows = []
    seed0 = config.RANDOM_SEED
    try:
        for seed in SEEDS:
            config.RANDOM_SEED = seed
            for name, refine in VARIANTS:
                config.REFINE = refine
                for t in config.TRAINS:
                    if (seed, name, t) in have:
                        continue
                    out, _, _ = train.run(t)
                    out.update({"정제": name, "시드": seed})
                    rows.append(out)
                    log.info("[시드 %d · %s] %s  PINN AUC %.3f 리프트1 %.1f | GBM AUC %.3f 리프트1 %.1f",
                             seed, name, t, out["PINN_전체_AUC"], out["PINN_전체_리프트1"],
                             out["GBM_전체_AUC"], out["GBM_전체_리프트1"])
                    # 중간 저장 — **기존 완료분을 보존**한다
                    pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(
                        raw_path, index=False, encoding="utf-8-sig")
    finally:
        config.RANDOM_SEED, config.REFINE = seed0, ("kf", "ae", "ekf")
    return pd.concat([done, pd.DataFrame(rows)], ignore_index=True)


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = [v[0] for v in VARIANTS]
    g = df.groupby(["계열", "정제"])[METRICS]
    summ = g.mean().round(4).add_suffix("_평균").join(g.std().round(4).add_suffix("_표준편차"))
    summ = summ.reset_index()
    summ["정제"] = pd.Categorical(summ["정제"], order, ordered=True)
    summ = summ.sort_values(["계열", "정제"])

    pairs = []
    for t, dt in df.groupby("계열"):
        piv = {m: dt.pivot(index="시드", columns="정제", values=m) for m in PAIR_METRICS}
        for a, b in itertools.combinations(order, 2):
            row = {"계열": t, "비교": f"{b} − {a}"}
            for m, sign in PAIR_METRICS.items():
                d = (piv[m][b] - piv[m][a]).dropna()
                better = int((sign * d > 0).sum())
                row[f"{m}_차이평균"] = round(float(d.mean()), 4)
                row[f"{m}_차이표준편차"] = round(float(d.std()), 4)
                row[f"{m}_개선시드"] = f"{better}/{len(d)}"
            pairs.append(row)
    return summ, pd.DataFrame(pairs)


def main() -> None:
    df = run_all()
    df.to_csv(config.OUTPUT_DIR / "refine_seeds_raw.csv", index=False, encoding="utf-8-sig")
    summ, pairs = summarize(df)
    summ.to_csv(config.OUTPUT_DIR / "refine_seeds_summary.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(config.OUTPUT_DIR / "refine_seeds_pairs.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 280)
    show = ["계열", "정제"] + [f"{m}_{s}" for m in
                              ("PINN_전체_AUC", "PINN_전체_리프트1", "PINN_전체_Brier개선%", "GBM_전체_AUC")
                              for s in ("평균", "표준편차")]
    print("\n=== 조합별 평균 ± 표준편차 (시드 %d개) ===" % len(SEEDS))
    print(summ[show].to_string(index=False))
    print("\n=== 짝비교 (같은 시드끼리 뺀 차이) ===")
    pc = ["계열", "비교"] + [f"{m}_{s}" for m in ("PINN_전체_AUC", "PINN_전체_리프트1", "PINN_전체_Brier개선%")
                            for s in ("차이평균", "개선시드")]
    print(pairs[pc].to_string(index=False))


if __name__ == "__main__":
    main()
