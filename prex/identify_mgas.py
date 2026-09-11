"""[진단] 밸브식 없이 m_gas·U·A 를 분리할 수 있는지 검증 — KF-PINN 설계 근거.

배경 (docs/kfpinn_state_space.md §3, htr31p_flow.md §7-2):
  P&ID로 히터↔정압기가 1:1이 아님이 확인돼 `gas_flow_proxy`(밸브식 역산)를 m_gas 로 쓸 수 없게 됐다.
  대안 경로가 있는지 두 식으로 확인한다 — 둘 다 **신뢰 태그(TI21Z/TI33P/TI-D2P)만** 쓴다.

  (A) ε-NTU 관계 — 매 분 직접 관측된다:
        ε   = (T_out − T_in)/(T_bath − T_in),   NTU = −ln(1−ε) = U·A/(m_gas·c_p)
      → **U·A 와 m_gas 의 "비"만 준다. 하나로는 분리 불가(구조적 미식별).**

  (B) 버너 OFF 수조 열수지 — 분리를 시도하는 두 번째 식:
        −M_w·c_w · dT_bath/dt = m_gas·c_p·(T_out − T_in) + Q_loss
      → y=−M_w·c_w·dT_bath/dt 를 x=c_p·(T_out−T_in) 에 회귀하면 기울기=m_gas, 절편=Q_loss.
      (A)의 NTU 와 합치면 U·A = NTU · m_gas · c_p 로 분리된다.

결론 (2026-09-11 실행): **(B)는 신뢰할 수 없다.** m_gas≈1.2 kg/s(설계의 0.09배),
  U·A≈1 kW/K(설계의 0.06배), R²≈0.38. 저부하 U·A 저하(Dittus-Boelter, 가스측 α∝m^0.8)로는
  16배 중 2.4배까지밖에 설명이 안 된다. 남은 괴리의 주범은 **진공식 잠열항**이다 —
  HTR-31P는 감압비등식이라 수조가 비등·응축하면 유효 열용량이 M_w·c_w(=현열 60.76 MJ/K)보다
  훨씬 커지고, dT_bath/dt 가 그만큼 작게 관측된다. (B)의 기울기·절편은 그 미지 배율로 함께
  나눠지므로 m_gas 와 Q_loss 가 동시에 과소 추정된다.

  → **그래서 m_gas 를 plug-in 추정값으로 넣지 않고 KF-PINN 잠재변수로 둔다.**
     동시에 수조 유효 열용량도 미지로 두어야 한다(잠열 배율). 상세: docs/kfpinn_state_space.md §4.

실행:  python -m prex.identify_mgas      (※ 진단용 — pipeline.main() 에는 안 엮여 있음)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .verify_ua import BURNER_ON_VALUE, burner_state_on_grid, load_events, load_trend

logger = logging.getLogger(__name__)

SETTLE_MIN = 10        # ON→OFF 전이 직후 과도구간 제거 [min]
MIN_DT_HEAT_C = 1.0    # 가스 승온폭이 이보다 작으면 신호가 약해 제외 [℃]
TRIM_FRAC = 0.05       # 회귀 잔차 상·하위 트리밍 비율(간이 robust)
MIN_YEAR_SAMPLES = 5_000


def build_off_balance_samples(trend: pd.DataFrame, burner: pd.Series) -> pd.DataFrame:
    """버너 OFF 구간에서 (x, y) = (c_p·ΔT_gas, −M_w c_w·dT_bath/dt) 샘플을 만든다."""
    t_in = trend[config.HTR31P_INLET_TAG]
    dt_s = trend.index.to_series().diff().dt.total_seconds()
    seg = trend["segment_id"]

    x = config.HTR_CP_GAS_J_KGK * (trend["TI33P"] - t_in)          # [W per kg/s]
    y = -config.HTR_MW_CW_J_PER_K * trend["TI-D2P"].diff() / dt_s  # [W] 수조가 잃는 열

    known = burner.notna()
    off = known & (burner != BURNER_ON_VALUE)
    new_run = (off != off.shift()) | (seg != seg.shift())
    within = trend.groupby(new_run.cumsum()).cumcount()

    valid = (
        off & off.shift(fill_value=False)
        & (dt_s == 60.0) & (seg == seg.shift())
        & (within >= SETTLE_MIN)
        & x.gt(config.HTR_CP_GAS_J_KGK * MIN_DT_HEAT_C)   # 가스가 실제로 데워지는 중
        & y.gt(0)                                          # 수조가 식는 중
        & x.notna() & y.notna()
    )
    logger.info("OFF 열수지 유효 샘플 %d행", int(valid.sum()))
    return pd.DataFrame({"x": x[valid], "y": y[valid]})


def fit_trimmed(x: np.ndarray, y: np.ndarray, trim: float = TRIM_FRAC) -> dict:
    """y = m·x + q 를 잔차 트리밍 2회로 적합. m=m_gas[kg/s], q=Q_loss[W]."""
    m = q = np.nan
    for _ in range(2):
        if len(x) < 100:
            break
        m, q = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
        r = y - (m * x + q)
        lo, hi = np.quantile(r, [trim, 1 - trim])
        keep = (r >= lo) & (r <= hi)
        x, y = x[keep], y[keep]
    resid = y - (m * x + q)
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan
    return {"m_gas_kg_s": m, "q_loss_w": q, "r2": r2, "n": len(x)}


def ua_from_ntu(trend: pd.DataFrame, mask: pd.Series, m_gas: float) -> tuple[float, float]:
    """ε의 중앙 NTU 와 m_gas 로 U·A 를 복원한다. Returns (ntu, ua_w_per_k)."""
    ntu = trend.loc[mask, "htx_ntu"].median()
    return ntu, ntu * m_gas * config.HTR_CP_GAS_J_KGK


def run() -> pd.DataFrame:
    trend = load_trend()
    if "htx_ntu" not in trend.columns:
        raise RuntimeError("htx_ntu 컬럼이 없습니다 — `python -m prex.pipeline` 재실행 필요.")
    burner = burner_state_on_grid(load_events(), trend.index)
    samples = build_off_balance_samples(trend, burner)

    pooled = fit_trimmed(samples["x"].to_numpy(), samples["y"].to_numpy())
    ntu_all, ua_all = ua_from_ntu(trend, trend["htx_ntu"].notna(), pooled["m_gas_kg_s"])
    logger.info(
        "[pooled] m_gas=%.2f kg/s (설계의 %.2f배) · Q_loss=%.1f kW · R²=%.3f | NTU=%.3f → U·A=%.2f kW/K (설계의 %.2f배)",
        pooled["m_gas_kg_s"], pooled["m_gas_kg_s"] / config.HTR_DESIGN_MGAS_KG_S,
        pooled["q_loss_w"] / 1e3, pooled["r2"],
        ntu_all, ua_all / 1e3, ua_all / config.HTR_UA_W_PER_K,
    )

    rows = []
    for year, idx in trend.groupby(trend.index.year).groups.items():
        s = samples.loc[samples.index.isin(idx)]
        if len(s) < MIN_YEAR_SAMPLES:
            continue
        fit = fit_trimmed(s["x"].to_numpy(), s["y"].to_numpy())
        in_year = pd.Series(trend.index.isin(idx), index=trend.index) & trend["htx_ntu"].notna()
        ntu, ua = ua_from_ntu(trend, in_year, fit["m_gas_kg_s"])
        rows.append({"year": year, **fit, "ntu_median": ntu, "ua_w_per_k": ua})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run()
    pd.set_option("display.width", 120)
    print("\n[연도별] 밸브식 없이 분리 시도한 m_gas / U·A")
    print(df.assign(
        m_gas=lambda d: d["m_gas_kg_s"].round(2),
        Q_loss_kW=lambda d: (d["q_loss_w"] / 1e3).round(1),
        UA_kW_K=lambda d: (d["ua_w_per_k"] / 1e3).round(2),
    )[["year", "n", "m_gas", "Q_loss_kW", "r2", "ntu_median", "UA_kW_K"]].to_string(index=False))
    print(f"\n설계값: m_gas {config.HTR_DESIGN_MGAS_KG_S:.2f} kg/s · U·A {config.HTR_UA_W_PER_K/1e3:.1f} kW/K")
    print("※ 이 결과가 설계와 크게 어긋나는 이유와 그로부터 나온 설계 결론은 모듈 docstring 참고.")
    out = config.OUTPUT_DIR / "identify_mgas_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
