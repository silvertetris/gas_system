"""유량계 없이 `m_tot` 을 역산한다 — 버너 가동률 열량법 (docs §20).

**없다는 사실 먼저**: Trend 175개월 전수 확인 결과 유량 계열 컬럼 40개(`FI61*`·`FQ61*`·`SG*`·`QY*`)는
**값이 하나도 없다**. `ZI41O/P/Q` 는 중앙 100·p10 0 의 사실상 개폐 신호라 밸브식 역산도 불가.
`PDI21x` 는 분해능 0.05·중앙 0. → **계열 유량은 어디에도 계측되지 않는다.**

**역산 원리**: 버너는 ON/OFF 사이클로 운전된다(`H31xOH`). 수조 온도가 정상상태인 창(window)에서는
축열 변화가 0 이므로 들어온 열 = 나간 열이다:

    s̄ · Q_b  =  Q_gas + Q_loss              s̄ = 그 창의 버너 ON 비율, Q_b = 버너 열출력

여기서 **`Q_loss` 는 따로 측정할 필요가 없다.** 그 히터가 `isolated`(가스 안 지나감) 이면서
정상상태인 창에서는 `Q_gas = 0` 이므로

    s̄_iso · Q_b = Q_loss                    ← 손실은 곧 '무부하 가동률'

두 식을 빼면 미지수가 사라진다:

    Q_gas = (s̄ − s̄_iso) · Q_b  =  m_i · c_p · Δ_i
    →  **m_i = (s̄ − s̄_iso) · Q_b / (c_p · Δ_i)**

`Q_b` 는 제작도서 설계값(P: 흡수열 0.894 MW ÷ 효율 0.904)이고, `s̄`·`s̄_iso`·`Δ_i` 는 전부 실측이다.
**ε 도 U·A 도 안 쓴다** — §19-4 에서 문제로 지목된 (H3) 역산을 우회한다.

그리고 이것이 (H5)의 독립 검증이 된다: (H5)는 `Σ m_i·Δ_i = m_tot·ΔH` 를 요구하므로
버너에서 나온 `m_i` 들로 `m_tot` 을 계산하면 **`Σ m_i ≤ m_tot` 이 성립해야** 한다.

⚠ 한계: `Q_b` 를 상수로 둔다. 실제 버너는 블로워 회전수로 변조되므로(문서 05 §3-3) `s̄` 는
점화시간 비율일 뿐 화력 비율이 아니다. → `m_i` 의 **절대값**은 `Q_b` 오차만큼 비례해 틀린다.
비율(`f_i`)과 시간 추세는 그 영향을 받지 않는다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import calorimetry, config, crosscheck

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

WINDOW_MIN = 60           # 창 길이 [분]
MIN_COVERAGE = 0.9        # 창 내 유효 데이터 비율
STATIONARY_DRIFT_C = 1.5  # 정상상태 판정: 창 양끝 수조온도 차이 [℃]
MIN_DELTA_C = 2.0
BURNER_EFF = 0.904        # 제작도서


def build_windows(g: pd.DataFrame, bn: dict[str, pd.Series]) -> pd.DataFrame:
    """창 단위 집계: 히터별 버너 가동률·수조 드리프트·상태·Δ, 계열별 ΔH."""
    idx = g.index.floor(f"{WINDOW_MIN}min")
    out = {}
    for train, units in config.TRAIN_UNITS.items():
        t_in = g[config.TRAIN_INLET[train]]
        out[f"dH_{train}"] = (g[f"t_hdr_{train}"] - t_in).groupby(idx).median()
        out[f"Tin_{train}"] = t_in.groupby(idx).median()
        for u in units:
            bath = g[f"TI-D2{u}"]
            out[f"s_{u}"] = bn[u].groupby(idx).mean()
            out[f"drift_{u}"] = bath.groupby(idx).last() - bath.groupby(idx).first()
            out[f"Tb_{u}"] = bath.groupby(idx).median()
            out[f"d_{u}"] = (g[f"TI33{u}"] - t_in).groupby(idx).median()
            r = g[f"regime_{u}"]
            out[f"reg_{u}"] = r.groupby(idx).agg(
                lambda s: s.iloc[0] if s.notna().all() and s.nunique() == 1 else None)
            out[f"cov_{u}"] = bath.notna().groupby(idx).mean()
    return pd.DataFrame(out)


def _unused_no_load_duty(W: pd.DataFrame, u: str, train: str) -> tuple[float, int, float]:
    """🔴 폐기 — 위 docstring 참고. 재현용으로만 남긴다."""
    """무부하 가동률.  `s̄_iso = b·(T_bath − T_amb)` — **절편 없는 1모수 모형**.

    손실은 주위와의 온도차에 비례하므로 `T_bath = T_amb` 에서 0 이어야 한다. 절편을 자유롭게 두면
    물리적으로 불가능한 해가 나온다(실측: B 호기가 `−0.385 + 0.00911·T_bath` 로 적합돼
    40℃ 에서 `s̄_iso < 0`). 주위온도 대용으로 입구 가스온도 `T_in` 을 쓴다 —
    두 계열 모두 지중 배관이라 연중 3~34℃ 로 외기를 따라간다(§3-1).
    """
    k = ((W[f"reg_{u}"] == "isolated") & (W[f"drift_{u}"].abs() < STATIONARY_DRIFT_C)
         & (W[f"cov_{u}"] > MIN_COVERAGE) & W[f"s_{u}"].notna() & W[f"Tb_{u}"].notna()
         & W[f"Tin_{train}"].notna())
    s = W[k]
    dt = (s[f"Tb_{u}"] - s[f"Tin_{train}"]).to_numpy(float)
    y = s[f"s_{u}"].to_numpy(float)
    ok = dt > 5.0
    if ok.sum() < 30:
        return np.nan, int(ok.sum()), np.nan
    b = float(np.linalg.lstsq(dt[ok, None], y[ok], rcond=None)[0][0])
    resid = y[ok] - b * dt[ok]
    ss = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else np.nan
    return b, int(ok.sum()), r2


def main() -> None:
    g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()
    bn = crosscheck.burner_states(g.index)
    W = build_windows(g, bn)
    log.info("창 %d개 (%d분)", len(W), WINDOW_MIN)

    cal = calorimetry.measure(g).set_index("히터")
    for u in config.UNITS:
        log.info("[%s] C_eff %.1f MJ/K, Q_loss %.1f kW (설계열량의 %.2f%%)",
                 u, cal.loc[u, "C_eff_MJK"], cal.loc[u, "Q_loss_kW"], cal.loc[u, "설계열량대비%"])

    rows = []
    for train, units in config.TRAIN_UNITS.items():
        qb = {u: config.design_duty_w(u) / BURNER_EFF for u in units}
        m = {}
        for u in units:
            q_loss = float(cal.loc[u, "Q_loss_kW"]) * 1e3
            q_in = (W[f"s_{u}"] * calorimetry.FIRE_FACTOR * qb[u] - q_loss).clip(lower=0.0)
            m[u] = q_in / (config.CP_GAS_J_KGK * W[f"d_{u}"])
            m[u] = m[u].where((W[f"reg_{u}"] == "flow") & (W[f"d_{u}"] > MIN_DELTA_C)
                              & (W[f"drift_{u}"].abs() < STATIONARY_DRIFT_C)
                              & (W[f"cov_{u}"] > MIN_COVERAGE))
            m[u] = m[u].where(W[f"reg_{u}"] != "isolated", 0.0)
        bwin = g[f"beta_{train}"].groupby(g.index.floor(f"{WINDOW_MIN}min")).median()
        mm = pd.DataFrame(m)
        dd = pd.DataFrame({u: W[f"d_{u}"] for u in units})
        dH = W[f"dH_{train}"]
        ok = mm.notna().all(axis=1) & dH.notna() & (dH > 1.0) & (mm.sum(axis=1) > 0)
        mtot = (mm * dd).sum(axis=1)[ok] / dH[ok]           # (H5) 로부터
        msum = mm.sum(axis=1)[ok]
        md = sum(config.design_mgas_kg_s(u) for u in units)
        rows.append({
            "계열": train, "히터": "+".join(units), "창수": int(ok.sum()),
            "m_tot 중앙_kgs": round(float(mtot.median()), 2),
            "설계비": round(float(mtot.median()) / md, 2),
            "p10~p90": f"{mtot.quantile(.1):.2f}~{mtot.quantile(.9):.2f}",
            "Σm_i 중앙": round(float(msum.median()), 2),
            "**Σm≤m_tot**": f"{100*float((msum <= mtot).mean()):.1f}%",
            "β=1−Σm/m_tot 중앙": round(float((1 - msum / mtot).median()), 3),
            "같은 창 전처리β 중앙": round(float(bwin[ok].median()), 3),
            "두 β 상관(Spearman)": round(float((1 - msum / mtot).corr(bwin[ok], method="spearman")), 3),
        })
        for u in units:
            mu = mm[u][ok]
            log.info("  [%s] m_%s 중앙 %.2f kg/s (설계 %.2f 의 %.2f배)",
                     train, u, mu.median(), config.design_mgas_kg_s(u),
                     mu.median() / config.design_mgas_kg_s(u))
    out = pd.DataFrame(rows)
    out.to_csv(config.OUTPUT_DIR / "backsolve_mtot.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 버너 가동률로 역산한 계열 유량 ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
