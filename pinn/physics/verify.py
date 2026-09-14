"""무모수 검정 — ε-NTU 형태가 함의하는 **독립성**을 직접 본다.

    ε = 1 − exp(−U·A/(m·c_p))

우변에 `T_bath` 도 `T_in` 도 없다. 즉 유량이 고정된 조건에서 **ε 은 수조온도·입구온도와
무관**해야 한다. 이건 자유 모수가 하나도 없는 예측이고, 틀리면 식이 틀린 것이다.

유량을 직접 못 보므로 **대리 조건(밸브개도·버너duty·입구온도·계절)으로 좁게 잘라** 구간
안에서 상관을 본다. 대리가 완벽하지 않으니 약한 잔여 상관은 예상된다 — 문제는 크기다.
`c_p`·`U` 자체가 온도에 약하게 의존하므로 0 을 기대하지도 않는다.

대조군: 같은 구간 나누기로 **무작위 치환한 ε** 의 상관 분포. 실측 상관이 그 분포와
구별되지 않으면 "의미 있는 의존성 없음"이다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import config, data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
RNG = np.random.default_rng(config.RANDOM_SEED)


def _key(cols: list[np.ndarray]) -> np.ndarray:
    """다차원 구간 인덱스를 정수 하나로 인코딩 (튜플 비교는 느리고 브로드캐스트가 깨진다)."""
    k = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        k = k * (int(c.max()) + 1) + c.astype(np.int64)
    return k


def _bins(s: pd.Series, k: int) -> np.ndarray:
    """분위 구간 인덱스. 값이 뭉쳐 있으면 실제 구간 수가 줄어든다(그대로 둔다)."""
    q = np.unique(np.nanquantile(s.to_numpy(dtype=float), np.linspace(0, 1, k + 1)))
    return np.searchsorted(q[1:-1], s.to_numpy(dtype=float), side="right")


def independence(d: pd.DataFrame, train: str) -> pd.DataFrame:
    """구간별 Spearman(ε, T_bath) 와 Spearman(ε, T_in). 대조군 동반."""
    units = config.TRAINS[train]["units"]
    rows = []
    for u in units:
        e = d[f"eps_{u}"]
        # 유량 대리 조건으로 구간 나누기
        b_valve = _bins(d["valve"], config.SLOPE_BINS["valve"])
        b_duty = _bins(d[f"duty_{u}"], config.SLOPE_BINS["duty"])
        b_tin = _bins(d["t_in"], config.SLOPE_BINS["t_in"])
        b_bath = _bins(d[f"bath_{u}"], config.SLOPE_BINS["t_in"])

        for target, cond in (("T_bath", [b_valve, b_duty, b_tin]),
                             ("T_in", [b_valve, b_duty, b_bath])):
            key = _key(cond)
            x = d[f"bath_{u}"] if target == "T_bath" else d["t_in"]
            obs, null, ns = [], [], []
            for k in pd.unique(key):
                m = key == k
                if m.sum() < config.SLOPE_MIN_N:
                    continue
                xi, ei = x[m].to_numpy(), e[m].to_numpy()
                if np.ptp(xi) < config.SLOPE_MIN_SPREAD_C:
                    continue
                # 큰 구간은 순위계산이 무거워 20만 행으로 자른다
                if len(xi) > 200_000:
                    s = RNG.choice(len(xi), 200_000, replace=False)
                    xi, ei = xi[s], ei[s]
                obs.append(stats.spearmanr(ei, xi)[0])
                null.append(stats.spearmanr(RNG.permutation(ei), xi)[0])
                ns.append(int(m.sum()))
            if not obs:
                continue
            obs, null = np.array(obs), np.array(null)
            rows.append({
                "계열": train, "히터": u, "대상": target, "구간수": len(obs),
                "행수중앙": int(np.median(ns)),
                "|ρ|중앙": round(float(np.median(np.abs(obs))), 3),
                "|ρ|p90": round(float(np.percentile(np.abs(obs), 90)), 3),
                "ρ중앙": round(float(np.median(obs)), 3),
                "대조군|ρ|중앙": round(float(np.median(np.abs(null))), 3),
                "|ρ|>0.2 구간비율": round(float((np.abs(obs) > 0.2).mean()), 3),
                "부호일관성": round(float(max((obs > 0).mean(), (obs < 0).mean())), 3),
            })
    return pd.DataFrame(rows)


def slope_check(d: pd.DataFrame, train: str) -> pd.DataFrame:
    """(Q3) 의 기울기 예측: ∂T_out/∂T_bath = ε, ∂T_out/∂T_in = 1 − ε.

    ⚠ ε 이 구간 안에서 `T_bath` 와 무관하면 이 기울기 일치는 **대수적으로 따라온다.**
    그래서 이 표는 `independence()` 의 결과를 온도 단위로 보여주는 **표현**이지
    독립된 증거가 아니다. 그렇게 읽어야 한다.
    """
    units = config.TRAINS[train]["units"]
    rows = []
    for u in units:
        key = _key([_bins(d["valve"], config.SLOPE_BINS["valve"]),
                    _bins(d[f"duty_{u}"], config.SLOPE_BINS["duty"]),
                    _bins(d["t_in"], config.SLOPE_BINS["t_in"])])
        sl, ep, ns = [], [], []
        for k in pd.unique(key):
            m = key == k
            if m.sum() < config.SLOPE_MIN_N:
                continue
            x, y = d[f"bath_{u}"][m].to_numpy(), d[f"out_{u}"][m].to_numpy()
            if np.ptp(x) < config.SLOPE_MIN_SPREAD_C:
                continue
            sl.append(np.polyfit(x, y, 1)[0])
            ep.append(float(d[f"eps_{u}"][m].mean()))
            ns.append(int(m.sum()))
        if not sl:
            continue
        sl, ep = np.array(sl), np.array(ep)
        rows.append({"계열": train, "히터": u, "구간수": len(sl), "행수중앙": int(np.median(ns)),
                     "기울기중앙": round(float(np.median(sl)), 3),
                     "ε중앙": round(float(np.median(ep)), 3),
                     "차이중앙": round(float(np.median(sl - ep)), 3),
                     "|차이|p90": round(float(np.percentile(np.abs(sl - ep), 90)), 3),
                     "상관(기울기,ε)": round(float(stats.spearmanr(sl, ep)[0]), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ind, slo = [], []
    for t in config.TRAINS:
        d = data.build(t)
        ind.append(independence(d, t))
        slo.append(slope_check(d, t))
    ind = pd.concat(ind, ignore_index=True)
    slo = pd.concat(slo, ignore_index=True)
    ind.to_csv(config.OUTPUT_DIR / "verify_independence.csv", index=False, encoding="utf-8-sig")
    slo.to_csv(config.OUTPUT_DIR / "verify_slope.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 검정 1: ε 은 수조온도·입구온도와 무관한가 (유량 대리 고정) ===")
    print(ind.to_string(index=False))
    print("\n=== 검정 1-b: 기울기 표현 (∂T_out/∂T_bath vs ε) ===")
    print(slo.to_string(index=False))


if __name__ == "__main__":
    main()
