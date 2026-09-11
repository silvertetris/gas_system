"""[진단] 버너 OFF 냉각곡선으로 U·A·Q_loss를 Q_burner 없이 식별할 수 있는지 검증.

목적 (docs/htr31p_flow.md §5-1, 06 §2-1 논의):
  "Q_burner 실측이 없으면 학습이 안 되나?"에 대한 데이터 기반 답을 낸다.
  버너가 꺼진 구간(H31POH=0)에서는 H1의 구동항이 사라지므로:

      M_w·c_w · dT_bath/dt = −U·A·(T_bath − T_gas) − Q_loss        (Q_burner=0)
   ⇒ dT_bath/dt = −(U·A / M_w·c_w)·(T_bath − T_gas) − Q_loss / M_w·c_w

  즉 dT_bath/dt 를 (T_bath − T_gas) 에 회귀하면
      기울기 a = −U·A / M_w·c_w    →  U·A   = −a · M_w·c_w
      절편   b = −Q_loss / M_w·c_w →  Q_loss = −b · M_w·c_w
  로 **Q_burner를 몰라도** U·A·Q_loss 를 뽑을 수 있다. M_w·c_w 는 제작도서 확보값(config).

판정 기준:
  (1) 식별성: pooled U·A_est 가 설계값(config.HTR_UA_W_PER_K, 16.2 kW/K)의 대략 0.5~2배 범위면
      "Q_burner 없이도 U·A가 데이터에서 식별된다"는 근거 → KF-PINN 열화추정 가능성 뒷받침.
  (2) 열화신호: 연도별 U·A_est 가 시간에 따라 하락하면 오염·튜브열화 지표가 실재.

⚠ 한계(결과 해석 시 반드시 감안):
  - 진공식(감압 비등) 잠열항은 이 현열 회귀에 안 들어감 → 비등 구간에선 유효 열용량이 커 보여 U·A 과대/과소 가능.
    (기본적으로 냉각 중 단조감소 구간만 쓰지만, PI-D2P로 비등여부 추가필터 권장 — FILTER_BOILING 참고)
  - OFF 중 가스가 계속 흐르는지(→ U·A 유효) vs 정지인지(→ ambient 손실만)에 따라 회귀가 잡는 대상이 달라짐.
  - dT_bath/dt 는 1분 차분이라 잡음이 큼 → 정착시간 제거 + 최소 |ΔT| + 잔차 트리밍으로 완화.
  - H31POH 값 해석(1=가동)이 반대면 BURNER_ON_VALUE 만 바꾸면 됨.

실행:  python -m prex.verify_ua      (※ 진단용 — pipeline.main() 에는 안 엮여 있음)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# --- 튜닝 파라미터 -----------------------------------------------------------
BURNER_ON_VALUE = 1.0     # H31POH 값이 이것이면 버너 ON (반대면 0.0 으로)
SETTLE_MIN = 10           # ON→OFF 전이 후 이 분(min)만큼은 과도구간이라 버림
MIN_ABS_DT = 2.0          # |T_bath − T_gas| 가 이보다 작으면 신호가 약해 제외 [℃]
MIN_RUN_MIN = SETTLE_MIN + 15   # OFF가 최소 이만큼 연속돼야 그 런을 쓴다 [min]
TRIM_FRAC = 0.05          # 회귀 잔차 상·하위 이 비율을 잘라 재적합(간이 robust)
FILTER_BOILING = False    # True 면 PI-D2P_norm 기준 비등 의심 구간 제외(아래 BOIL_* 사용)
BOIL_PI_D2P_NORM_MAX = 0.0  # PI-D2P_norm 이 이보다 크면(진공 얕음=고압쪽) 비등 의심 → 제외


# --- 데이터 로드 -------------------------------------------------------------
def load_trend() -> pd.DataFrame:
    """1분 그리드 trend 을 Time 인덱스로 반환. parquet 있으면 재사용, 없으면 파이프라인 실행."""
    p = config.OUTPUT_DIR / "trend_htr31p.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        return df.set_index("Time")
    logger.info("trend parquet 없음 → run_trend_pipeline() 실행")
    from .trend import run_trend_pipeline
    return run_trend_pipeline()


def load_events() -> pd.DataFrame:
    """정제된 전체 알람 이벤트(long, SET/RESET 모두 포함). parquet 있으면 재사용."""
    p = config.OUTPUT_DIR / "alarm_events_htr31p.parquet"
    if p.exists():
        return pd.read_parquet(p)
    logger.info("alarm parquet 없음 → run_alarm_pipeline() 실행")
    from .alarm import run_alarm_pipeline
    long, _ = run_alarm_pipeline()
    return long


def burner_state_on_grid(events: pd.DataFrame, grid_index: pd.DatetimeIndex) -> pd.Series:
    """H31POH SET/RESET 이벤트를 1분 그리드에 forward-fill → 각 시각의 버너 상태.

    Returns: grid_index 를 인덱스로 하는 Series {1.0=ON, 0.0=OFF, NaN=첫 이벤트 이전(불명)}.
    """
    h = events.loc[events["tag"] == "H31POH", ["Time", "value"]].dropna()
    if h.empty:
        raise RuntimeError("H31POH 이벤트가 없습니다 — 버너 상태 복원 불가.")
    h = h.sort_values("Time").drop_duplicates("Time", keep="last")
    left = pd.DataFrame({"Time": grid_index})
    merged = pd.merge_asof(left, h, on="Time", direction="backward")  # 직전 상태 전파
    state = pd.Series(merged["value"].to_numpy(), index=grid_index, name="burner_raw")
    logger.info("H31POH 이벤트 %d건 → ON비율(known기준) %.1f%%",
                len(h), 100 * (state == BURNER_ON_VALUE).sum() / max(state.notna().sum(), 1))
    return state


# --- 냉각 샘플 준비 ----------------------------------------------------------
def build_off_samples(trend: pd.DataFrame, burner_raw: pd.Series) -> pd.DataFrame:
    """버너 OFF 연속구간에서 (dT_bath/dt, ΔT=T_bath−T_gas) 유효 샘플을 만든다.

    - T_gas = ½·(TI21Z + TI33P)  (입·출구 평균; 입구는 P호기 계열인 TI21Z 단독 — config 주석 참조)
    - 정착시간(SETTLE_MIN) 제거, 최소 런길이/최소 |ΔT|/segment 경계/결측 처리 포함.
    """
    df = trend.copy()
    t_bath = df["TI-D2P"]
    t_in = df[config.HTR31P_INLET_TAG]
    t_gas = 0.5 * (t_in + df["TI33P"])
    dt_s = df.index.to_series().diff().dt.total_seconds()          # 그리드 간격(초), 보통 60
    dTb_dt = t_bath.diff() / dt_s                                  # dT_bath/dt [℃/s]
    dT_pair = 0.5 * ((t_bath - t_gas) + (t_bath - t_gas).shift(1))  # 구간 평균 ΔT

    known = burner_raw.notna()
    off = known & (burner_raw != BURNER_ON_VALUE)                  # OFF(=known 이면서 ON 아님)

    # OFF 연속 런 식별 (OFF 토글 또는 segment 변경 시 새 런)
    seg = df["segment_id"]
    new_run = (off != off.shift()) | (seg != seg.shift())
    run_id = new_run.cumsum()
    within = df.groupby(run_id).cumcount()                        # 런 내 0-based 분 인덱스
    run_len = df.groupby(run_id)["TI-D2P"].transform("size")

    valid = (
        off & off.shift(fill_value=False)                         # 양 끝 모두 OFF
        & (dt_s == 60.0)                                          # 연속 1분
        & (seg == seg.shift())                                    # 같은 segment
        & dTb_dt.notna() & dT_pair.notna()
        & (within >= SETTLE_MIN)                                  # 과도구간 제거
        & (run_len >= MIN_RUN_MIN)                                # 충분히 긴 OFF 런만
        & (dT_pair.abs() >= MIN_ABS_DT)                           # 신호 약한 것 제외
    )
    if FILTER_BOILING and "PI-D2P_norm" in df.columns:
        valid &= df["PI-D2P_norm"] <= BOIL_PI_D2P_NORM_MAX        # 비등 의심 구간 제외

    out = pd.DataFrame({
        "dTbath_dt": dTb_dt[valid],       # [℃/s]
        "dT": dT_pair[valid],             # [℃]  (T_bath − T_gas)
        "year": df.index[valid].year,
        "era": df["PI-D2P_era"][valid] if "PI-D2P_era" in df.columns else np.nan,
    })
    logger.info("OFF 유효 샘플 %d개 (전체 %d분 중)", len(out), len(df))
    return out


# --- 회귀(간이 robust) -------------------------------------------------------
def fit_cooling(dT: np.ndarray, dTdt: np.ndarray, trim: float = TRIM_FRAC) -> dict:
    """dTdt = a·dT + b 적합 → U·A, Q_loss 환산. 잔차 트리밍으로 이상치 완화.

    OFF일 때 dTdt < 0, dT > 0 이 정상(수조가 가스/주변보다 뜨거워 식음) → a<0, U·A>0 기대.
    """
    mc = config.HTR_MW_CW_J_PER_K
    dT = np.asarray(dT, float); dTdt = np.asarray(dTdt, float)
    n0 = len(dT)
    if n0 < 30:
        return {"n": n0, "note": "샘플 부족"}

    def _ols(x, y):
        A = np.vstack([x, np.ones_like(x)]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = a * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or np.nan
        return a, b, 1.0 - ss_res / ss_tot

    a, b, r2 = _ols(dT, dTdt)
    if trim and n0 > 100:                                   # 잔차 상·하위 trim% 잘라 재적합
        resid = dTdt - (a * dT + b)
        lo, hi = np.quantile(resid, [trim, 1 - trim])
        m = (resid >= lo) & (resid <= hi)
        a, b, r2 = _ols(dT[m], dTdt[m])
        n_used = int(m.sum())
    else:
        n_used = n0

    result = {
        "n": n0, "n_used": n_used, "slope_a": a, "intercept_b": b, "r2": r2,
        "UA_W_per_K": -a * mc, "Q_loss_W": -b * mc,
    }
    # scipy 있으면 Theil-Sen robust 기울기로 교차확인 (없으면 생략)
    try:
        from scipy import stats
        ts = stats.theilslopes(dTdt, dT)
        result["UA_W_per_K_theilsen"] = -ts[0] * mc
    except Exception:
        pass
    return result


# --- 리포트 ------------------------------------------------------------------
def run() -> pd.DataFrame:
    """전체 검증 실행. pooled + 연도별 U·A_est 표를 출력하고 요약 DataFrame 반환."""
    trend = load_trend()
    events = load_events()
    burner_raw = burner_state_on_grid(events, trend.index)
    samples = build_off_samples(trend, burner_raw)

    design_ua = config.HTR_UA_W_PER_K
    print("\n===== 버너 OFF 냉각곡선 → U·A 식별 검증 =====")
    print(f"설계 U·A = {design_ua:,.0f} W/K ({design_ua/1000:.2f} kW/K), "
          f"M_w·c_w = {config.HTR_MW_CW_J_PER_K/1e6:.2f} MJ/K")

    pooled = fit_cooling(samples["dT"].to_numpy(), samples["dTbath_dt"].to_numpy())
    if "UA_W_per_K" in pooled:
        ratio = pooled["UA_W_per_K"] / design_ua
        verdict = "✅ 식별 성공(설계 0.5~2배)" if 0.5 <= ratio <= 2.0 else "⚠️ 설계와 괴리 — 한계항목 점검"
        print(f"\n[POOLED] n={pooled['n']:,}(사용 {pooled['n_used']:,}), R²={pooled['r2']:.3f}")
        print(f"  U·A_est = {pooled['UA_W_per_K']:,.0f} W/K "
              f"(설계 대비 {ratio:.2f}배) → {verdict}")
        print(f"  Q_loss_est = {pooled['Q_loss_W']:,.0f} W")
        if "UA_W_per_K_theilsen" in pooled:
            print(f"  (Theil-Sen 교차확인 U·A = {pooled['UA_W_per_K_theilsen']:,.0f} W/K)")
    else:
        print(f"\n[POOLED] {pooled.get('note', '적합 불가')}")

    # 연도별 = 열화 추세 (하락하면 오염/튜브열화 신호)
    rows = []
    print("\n[연도별] U·A_est 추세 (하락 = 열화 신호):")
    print(f"  {'year':>6} {'n_used':>8} {'R²':>6} {'U·A[kW/K]':>10} {'Q_loss[kW]':>11}")
    for year, g in samples.groupby("year"):
        r = fit_cooling(g["dT"].to_numpy(), g["dTbath_dt"].to_numpy())
        if "UA_W_per_K" not in r:
            continue
        rows.append({"year": int(year), "n_used": r["n_used"], "r2": r["r2"],
                     "UA_kW_per_K": r["UA_W_per_K"] / 1000, "Q_loss_kW": r["Q_loss_W"] / 1000})
        print(f"  {int(year):>6} {r['n_used']:>8,} {r['r2']:>6.3f} "
              f"{r['UA_W_per_K']/1000:>10.2f} {r['Q_loss_W']/1000:>11.2f}")

    summary = pd.DataFrame(rows)
    out_path = config.OUTPUT_DIR / "verify_ua_summary.csv"
    if not summary.empty:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"\n연도별 요약 저장: {out_path}")
    print("\n※ 해석 한계는 모듈 docstring 참고(진공 잠열항·OFF중 가스흐름·차분 잡음).")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
