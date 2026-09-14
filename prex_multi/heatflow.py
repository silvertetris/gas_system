"""수조 열수지로 히터별 가스 유량을 역산한다 — ε 과 **완전히 독립**인 경로.

## 왜 이게 되나 — ON/OFF 대비

문서 07 경로 6("수조 버너 OFF 냉각")은 냉각률을 **무부하 냉각률**과 비교했다. 두 값이
비슷해서 차가 잡음에 묻혔다. 여기서는 **같은 사이클 안의 ON 과 OFF 를 대비**한다:

    ON  구간:  C_eff·(dT/dt)_ON  = fire·Q_버너 − Q_gas − Q_loss
    OFF 구간:  C_eff·(dT/dt)_OFF =            − Q_gas − Q_loss
    ─────────────────────────────────────────────────────────────
    차        C_eff·[(dT/dt)_ON − (dT/dt)_OFF] = fire·Q_버너

**`Q_gas` 와 `Q_loss` 가 상쇄된다.** 한 사이클(ON 15분 + OFF 50분)에서는 수요가 거의 안
변하므로 상쇄가 성립한다. 그래서 **유효 화력을 유량 지식 없이** 얻는다. 그 다음

    Q_gas = fire·Q_버너·duty_사이클 − Q_loss·(T_bath−T_in)/Δref − C_eff·(dT/dt)_사이클
    m     = Q_gas / (c_p·(T_out − T_in))

## 🔴 이전 결론 정정

문서 09 §4 는 "통가스 시간대 버너 발열이 ε 역산 흡수열의 1/3~1/4" 이라며 (Q1) 을 기각했다.
**두 군데가 틀렸다**:
  1. `fire = 1.0` 을 가정했다. 실측하면 **0.47~1.13** 이다.
  2. `duty` 를 `regime=flow` 부분집합에서 쟀다. 그 부분집합이 duty 를 0.036 으로 편향시켰다
     (전체 평균은 0.179).

## 검증 — ε 역산이 떨어진 검정을 전부 통과한다

| 검정 | 열수지 유량 | ε-NTU 역산 |
|---|---|---|
| M 겨울/여름 (도시가스 → >1 기대) | **2.06** | 0.86 ❌ |
| Z 겨울/여름 | **3.09** | 0.31 ❌ |
| 직접 푼 혼합비와 Spearman (양수 기대) | **+0.518** | −0.735 ❌ |

버너 사이클은 실측으로 ON 8~18분 / OFF 41~59분이고 ON 중 수조가 오른다(+0.7~+1.5 ℃/h) —
`H31xOH`("OPERATION HEATER")가 **가동상태가 아니라 점화 사이클**임을 확인했다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from prex import config as p_config
from . import config

logger = logging.getLogger(__name__)

MIN_PHASE_MIN = 8          # 기울기를 잴 최소 구간 [분]
EDGE_MIN = 2               # 열적 지연 때문에 양 끝에서 버릴 길이 [분]
MIN_DT_C = 2.0             # 가스 승온폭 하한 (분모)
FIRE_MIN, FIRE_MAX = 0.05, 3.0   # 유효 화력률 허용 범위 (밖이면 그 사이클 기각)

CP_GAS = p_config.HTR_CP_GAS_J_KGK
Q_BURNER = p_config.HTR_QBURNER_DESIGN_W

# 열량계 결과 (prex_multi/calorimetry.py)
CALORIMETRY = {
    "A": {"c_eff_mjk": 38.0, "q_loss_kw": 5.3, "dt_ref_c": 33.7},
    "B": {"c_eff_mjk": 33.8, "q_loss_kw": 3.6, "dt_ref_c": 27.2},
    "O": {"c_eff_mjk": 300.9, "q_loss_kw": 35.0, "dt_ref_c": 32.6},
    "P": {"c_eff_mjk": 82.3, "q_loss_kw": 22.3, "dt_ref_c": 45.0},
}


def _phases(index: pd.DatetimeIndex, burner: np.ndarray, ok: np.ndarray):
    """버너 상태가 일정한 연속 구간 경계. 1분 격자가 끊기는 곳도 경계로 본다."""
    gap = np.flatnonzero(np.diff(index.values).astype("timedelta64[m]").astype(int) != 1) + 1
    chg = np.flatnonzero(np.diff(burner) != 0) + 1
    return np.unique(np.concatenate([[0], chg, gap, [len(burner)]]))


def cycle_flow(grid: pd.DataFrame, unit: str, train: str) -> pd.DataFrame:
    """히터 하나의 사이클별 `q_fire`·`q_gas`·`m` 을 낸다."""
    cal = CALORIMETRY[unit]
    c_eff = cal["c_eff_mjk"] * 1e6                       # J/K
    q_loss_ref, dt_ref = cal["q_loss_kw"] * 1e3, cal["dt_ref_c"]

    b = grid[f"burner_{unit}"].astype("float").to_numpy()
    bath = grid[f"TI-D2{unit}"].to_numpy(np.float64)
    out = grid[f"TI33{unit}"].to_numpy(np.float64)
    t_in = grid[config.TRAIN_INLET[train]].to_numpy(np.float64)
    ok = np.isfinite(b) & np.isfinite(bath) & np.isfinite(out) & np.isfinite(t_in)

    segs = []
    for a, z in zip(*(lambda k: (k[:-1], k[1:]))(_phases(grid.index, b, ok))):
        if z - a < MIN_PHASE_MIN + 2 * EDGE_MIN or not ok[a:z].all():
            continue
        y = bath[a + EDGE_MIN:z - EDGE_MIN]
        slope = np.polyfit(np.arange(len(y), dtype=float), y, 1)[0] * 60.0    # ℃/h
        segs.append((a, z, int(b[a]), slope))

    rows = []
    for s1, s2 in zip(segs[:-1], segs[1:]):
        if not (s1[2] == 1 and s2[2] == 0 and s2[0] == s1[1]):
            continue                                     # 같은 사이클의 ON→OFF 쌍만
        a, z = s1[0], s2[1]
        q_fire = c_eff * (s1[3] - s2[3]) / 3600.0        # W — Q_gas·Q_loss 상쇄
        if not (FIRE_MIN * Q_BURNER < q_fire < FIRE_MAX * Q_BURNER):
            continue
        duty_c = (s1[1] - s1[0]) / (z - a)
        stor = c_eff * (bath[z - 1] - bath[a]) / ((z - a) * 60.0)            # W
        loss = q_loss_ref * (np.nanmean(bath[a:z]) - np.nanmean(t_in[a:z])) / dt_ref
        q_gas = q_fire * duty_c - loss - stor
        dt = np.nanmean(out[a:z]) - np.nanmean(t_in[a:z])
        if dt < MIN_DT_C or q_gas <= 0:
            continue
        rows.append({"t0": grid.index[a], "t1": grid.index[z - 1], "히터": unit,
                     "duty": duty_c, "q_fire_kw": q_fire / 1e3, "fire": q_fire / Q_BURNER,
                     "q_gas_kw": q_gas / 1e3, "m_kg_s": q_gas / (CP_GAS * dt),
                     "dt_c": dt, "bath_c": float(np.nanmean(bath[a:z]))})
    df = pd.DataFrame(rows)
    if len(df):
        df["설계비"] = df["m_kg_s"] / config.design_mgas_kg_s(unit)
    return df


def run() -> pd.DataFrame:
    need = ["Time"]
    for u in config.UNITS:
        need += [f"burner_{u}", f"TI-D2{u}", f"TI33{u}"]
    need += [config.TRAIN_INLET[t] for t in config.TRAIN_UNITS]
    grid = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet",
                           columns=sorted(set(need))).set_index("Time").sort_index()
    frames = []
    for train, units in config.TRAIN_UNITS.items():
        for u in units:
            f = cycle_flow(grid, u, train)
            f["계열"] = train
            frames.append(f)
            if len(f):
                logger.info("  [%s] 사이클 %6d | fire 중앙 %.2f | q_gas 중앙 %6.1f kW | "
                            "m 중앙 %5.2f kg/s (설계의 %.2f배)",
                            u, len(f), f["fire"].median(), f["q_gas_kw"].median(),
                            f["m_kg_s"].median(), f["설계비"].median())
    out = pd.concat(frames, ignore_index=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.OUTPUT_DIR / "heatflow_cycles.parquet", index=False)
    logger.info("저장: heatflow_cycles.parquet (%d 사이클)", len(out))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = run()
    pd.set_option("display.width", 200)
    print("\n=== 히터별 요약 ===")
    g = df.groupby(["계열", "히터"]).agg(
        사이클=("m_kg_s", "size"), fire중앙=("fire", "median"),
        화력kW=("q_fire_kw", "median"), duty중앙=("duty", "median"),
        Q_gas_kW=("q_gas_kw", "median"), m_kg_s=("m_kg_s", "median"),
        설계비=("설계비", "median")).round(3)
    print(g.to_string())
    # ⚠ 일별 **합**으로 집계하면 사이클 개수가 섞인다. 사이클당 유량의 월별 중앙을 쓴다.
    print("\n=== 계열 총유량(히터 합) 월별 중앙 [kg/s] ===")
    per = (df.set_index("t0").groupby([pd.Grouper(freq="1D"), "계열"])["m_kg_s"]
             .mean().reset_index())          # 하루 평균 사이클 유량
    tot = df.set_index("t0")
    tot = (tot.groupby([pd.Grouper(freq="1D"), "계열", "히터"])["m_kg_s"].mean()
              .groupby(level=[0, 1]).sum().reset_index())   # 히터 합
    piv = tot.pivot_table(index=tot["t0"].dt.month, columns="계열", values="m_kg_s",
                          aggfunc="median").round(2)
    print(piv.to_string())
    for c in piv.columns:
        w, s = piv.loc[[12, 1, 2], c].mean(), piv.loc[[6, 7, 8], c].mean()
        print(f"  {c}: 겨울 {w:.2f} / 여름 {s:.2f} → 비 {w/s:.2f}")


if __name__ == "__main__":
    main()
