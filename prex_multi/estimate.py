"""4대 히터 m_gas · Q_loss **분리** 추정 + 정비 주기별 건전도 φ.

왜 분리하나 (docs/htr31p_flow.md §11~§13 의 시행착오 끝에 도달한 구조):
  버너 OFF 수조 열수지  −C_eff·dT_bath/dt = m_gas·c_p·ΔT_gas + Q_loss − Q_잔열
  에서 두 미지수를 **한 회귀로 풀면 안 된다.** 두 조건이 서로 충돌하기 때문이다:

  - `Q_loss`(절편)는 **긴 OFF**, 즉 노내 잔열이 다 빠진 뒤에만 깨끗하다.
    실측(A호기, 경과분 구간별 절편): −5.1 → −8.4 → −4.6 → −3.0 → **+2.3 → +4.6 kW**.
    부호가 60분 근방에서 뒤집힌다. 그 전 구간을 쓰면 절편이 음수(=물리적으로 불가)로 끌린다.
  - `m_gas`(기울기)는 **짧은 OFF** 에서만 신호가 산다. 긴 OFF 는 곧 "가스 수요가 없는 시간"
    이라 실제 유량 자체가 낮다(실측: 480분+ 구간에서 설계의 0.07배로 붕괴). 선택 편향이다.

  → **서로 다른 표본으로 가른다.** 긴 OFF 로 Q_loss 를 먼저 확정하고, 그 값을 **고정**한 채
     짧은 OFF 에서 m_gas 를 재추정한다. 두 미지수를 한 설계행렬에 넣지 않으므로
     §11-2 에서 추정을 깨뜨렸던 조건수 문제(cond=407.8, C_eff 가 음수로 나옴)도 사라진다.

⚠ 양자화: `TI-D2x` 분해능이 0.061℃다. 인접 차분은 대부분 0 아니면 ±1양자라 dT/dt 가
  계단이 된다(§"m_gas 시변" 오판의 원인). 단순 차분 대신 **OFF 런 전체를 시간에 대해
  최소제곱 회귀**해 기울기 하나를 뽑는다. 런당 표본이 20분 이상이라 양자 계단이 평균된다.
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd

from prex import config as p_config

from . import config

logger = logging.getLogger(__name__)

LONG_OFF_MIN = 240      # Q_loss 용 하한 [분] — 잔열이 빠진 뒤
SHORT_OFF_MAX = 120     # m_gas 용 상한 [분] — 부하가 아직 살아 있는 구간
SETTLE_MIN = 10         # OFF 직후 버림 [분]
MIN_RUN_LEN = 20        # 기울기 회귀에 필요한 최소 표본 [분]
MIN_BATH_C = 20.0       # 계측 드롭아웃(0℃)·완전정지 제외
MIN_GAS_DT = 2.0        # 가스 승온폭 하한 [℃] — 통가스 중임을 보장
MIN_RUNS = 50           # 회귀 최소 런 수
TRIM = 0.05             # 양쪽 5% 잔차 절사(2회)


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------
def load_all() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Trend 격자 + 히터별 버너 가동상태(H31xOH)를 같은 인덱스로 정렬해 반환."""
    tr = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()
    burner: dict[str, pd.Series] = {}
    for fkey, tags in [("DI_06", ["H31AOH", "H31BOH", "H31OOH"]), ("DI_07", ["H31POH"])]:
        path = [x for x in glob.glob(str(config.ALARM_DI_DIR / "*.csv")) if fkey in x][0]
        d = pd.read_csv(path, usecols=["Time"] + tags)
        d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
        d = d.dropna(subset=["Time"])
        for t in tags:
            s = (d[["Time", t]].dropna().sort_values("Time")
                 .drop_duplicates("Time", keep="last"))
            # DI 는 상태변화 시점만 기록되는 이벤트 로그다 → 직전 값으로 채운다(backward asof).
            m = pd.merge_asof(pd.DataFrame({"Time": tr.index}), s, on="Time", direction="backward")
            burner[t[3]] = pd.Series(m[t].to_numpy(), index=tr.index)
    return tr, burner


# ---------------------------------------------------------------------------
# OFF 런 요약 (벡터화)
# ---------------------------------------------------------------------------
def off_runs(tr: pd.DataFrame, burner: pd.Series, u: str) -> pd.DataFrame:
    """버너 OFF 런마다 (길이, dT_bath/dt 기울기, ΔT_gas 중앙값) 한 줄.

    기울기는 런 내부 최소제곱: slope = Σ(t−t̄)(T−T̄) / Σ(t−t̄)². 양자화 대응.
    """
    bath = tr[f"TI-D2{u}"]
    ok = (burner == 0) & bath.notna() & tr[f"dT_{u}"].notna() & (bath > MIN_BATH_C)
    # 세그먼트(장기 결측으로 끊긴 구간)를 넘어가는 런은 만들지 않는다.
    run = ((ok != ok.shift()) | (tr["segment_id"] != tr["segment_id"].shift())).cumsum()

    f = pd.DataFrame({"run": run[ok], "T": bath[ok].to_numpy(),
                      "dt_gas": tr.loc[ok, f"dT_{u}"].to_numpy()})
    pos = f.groupby("run").cumcount()
    f = f[pos >= SETTLE_MIN]                       # OFF 직후 과도구간 버림
    t = f.groupby("run").cumcount().to_numpy() * 60.0   # 1분 격자 → 초
    f["t"] = t

    g = f.groupby("run")
    n = g.size()
    tb, Tb = g["t"].mean(), g["T"].mean()
    stt = g["t"].apply(lambda s: float(((s - s.mean()) ** 2).sum()))
    # Σ(t−t̄)(T−T̄) = Σ t·T − n·t̄·T̄
    stT = (f.assign(tT=f["t"] * f["T"]).groupby("run")["tT"].sum() - n * tb * Tb)
    out = pd.DataFrame({
        "minutes": n,
        "slope": stT / stt.replace(0.0, np.nan),
        "dt_gas": g["dt_gas"].median(),
    })
    return out[(out["minutes"] >= MIN_RUN_LEN) & (out["dt_gas"] > MIN_GAS_DT)
               & out["slope"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 절사 최소제곱
# ---------------------------------------------------------------------------
def _trim_fit(x: np.ndarray, y: np.ndarray, fix_intercept: float | None = None):
    """잔차 양쪽 5% 를 두 번 절사하며 적합. fix_intercept 가 있으면 기울기만 푼다."""
    slope = intercept = np.nan
    for _ in range(3):
        if len(x) < MIN_RUNS:
            break
        if fix_intercept is None:
            slope, intercept = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
        else:
            intercept = fix_intercept
            slope = float(np.linalg.lstsq(x[:, None], y - intercept, rcond=None)[0][0])
        r = y - (slope * x + intercept)
        lo, hi = np.quantile(r, [TRIM, 1 - TRIM])
        keep = (r >= lo) & (r <= hi)
        x, y = x[keep], y[keep]
    if len(y) < 2 or not np.isfinite(slope):
        return slope, intercept, np.nan, len(x)
    r = y - (slope * x + intercept)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum(r ** 2) / ss) if ss > 0 else np.nan
    return float(slope), float(intercept), r2, len(x)


def estimate(runs: pd.DataFrame, c_eff: float) -> dict:
    """1) 긴 OFF → Q_loss,  2) Q_loss 고정 → 짧은 OFF 에서 m_gas."""
    cp = config.CP_GAS_J_KGK
    lg = runs[runs["minutes"] >= LONG_OFF_MIN]
    sh = runs[runs["minutes"] <= SHORT_OFF_MAX]
    res = {"n_long": len(lg), "n_short": len(sh), "m_gas": np.nan, "q_loss_kw": np.nan,
           "r2_long": np.nan, "r2_short": np.nan}
    if not np.isfinite(c_eff) or len(lg) < MIN_RUNS or len(sh) < MIN_RUNS:
        return res
    _, q_loss, r2l, _ = _trim_fit((cp * lg["dt_gas"]).to_numpy(float),
                                  (-c_eff * lg["slope"]).to_numpy(float))
    m_gas, _, r2s, _ = _trim_fit((cp * sh["dt_gas"]).to_numpy(float),
                                 (-c_eff * sh["slope"]).to_numpy(float),
                                 fix_intercept=q_loss)
    res.update(m_gas=m_gas, q_loss_kw=q_loss / 1000.0, r2_long=r2l, r2_short=r2s)
    return res


# ---------------------------------------------------------------------------
# 유효 열용량 / 건전도
# ---------------------------------------------------------------------------
def c_eff_from_ramps(tr: pd.DataFrame, burner: pd.Series, u: str) -> tuple[float, int]:
    """자유 승온(냉간 기동) 구간에서 C_eff = Q_흡수 / (dT_bath/dt).

    버너 입열은 설계 흡수열 ÷ 효율 0.904(제작도서). ON 비율로 가중한다.
    """
    q_in = config.design_duty_w(u) / 0.904
    T = tr[f"TI-D2{u}"]
    W = 180  # 3시간 창
    rising = (T.diff(W) > 10).fillna(False)
    grp = (rising != rising.shift()).cumsum()
    vals = []
    for _, g in T[rising].groupby(grp[rising]):
        a = g.index.min() - pd.Timedelta(minutes=W)
        b = g.index.max()
        if a < T.index.min():
            continue
        Ta, Tb = T.get(a, np.nan), T.get(b, np.nan)
        if not (np.isfinite(Ta) and np.isfinite(Tb)):
            continue
        dT, dur = float(Tb - Ta), (b - a).total_seconds()
        if dT < 15 or dur < 3600:
            continue
        on = float((burner.loc[a:b] == 1).mean())
        if on < 0.3:
            continue
        vals.append(q_in * on / (dT / dur))
    return (float(np.median(vals)) if vals else np.nan), len(vals)


def ua_clean(u: str, m_gas: float) -> float:
    """직렬 열저항 청정 U·A [W/K]:  1/U = 1/(α_io·(m/m_d)^0.8) + 1/α_o.

    지수 0.8(Dittus-Boelter)·수조측 유량무관은 남사/안중 제작도서 부하별 U 표로 검증됨(§12-2, 오차 −0.1%).
    면적은 설계유량비 스케일(T3 프록시, config.scaled_ua_w_per_k 와 같은 근거).
    """
    if not np.isfinite(m_gas) or m_gas <= 0:
        return np.nan
    ratio = m_gas / config.design_mgas_kg_s(u)
    a_io = p_config.HTR_ALPHA_IO_KCAL * ratio ** p_config.GAS_SIDE_EXPONENT
    u_kcal = 1.0 / (1.0 / a_io + 1.0 / p_config.HTR_ALPHA_O_KCAL)
    area = config.P_AREA_M2 * config.design_mgas_kg_s(u) / config.design_mgas_kg_s("P")
    return u_kcal * area * config.KCAL_TO_J / 3600.0


# ---------------------------------------------------------------------------
# 기각된 접근 2가지 — 왜 안 되는지 (다시 시도하지 않도록 남긴다)
# ---------------------------------------------------------------------------
# (기각 1) "긴 OFF 로 Q_loss, 짧은 OFF 로 m_gas" 분리.
#   런 길이별 자유적합의 절편이 이렇게 감쇠한다(A호기, kW):
#       0~30분 234 → 30~60 147 → 60~120 103 → 120~240 55 → 240~480 11 → 480분+ 7.9
#   240분에도 11 kW 가 남으므로 "240분이면 깨끗하다"는 전제가 깨진다. 그 값을 고정해
#   짧은 OFF 에 적용하면 수백 kW 가 전부 m_gas 로 흡수된다(그래서 설계의 0.6~0.9배라는
#   '그럴듯한' 숫자가 나왔다 — 신뢰할 수 없다).
#
# (기각 2) 잔열을 Q0·exp(−t/τ) 로 놓고 (m_gas, Q_loss, Q0, τ) 동시 추정.
#   R² 가 0.99 로 치솟지만 **Q0 가 음수**(A: −1,254 kW, O: −18,001 kW = 설계열량의 10배)로
#   나온다. 물리적으로 불가능하다. 원인: τ 가 격자 하한(10분)에 붙으면 ḡ(L;τ) ≈ (τ/L)·상수라
#   사실상 1/L 회귀항이 되고, "짧은 런일수록 빨리 식는다"를 그대로 재현할 뿐이다.
#
#   → 그런데 이 실패가 핵심을 알려준다. 짧은 런에서 빨리 식는 이유는 잔열이 아니라
#     **런 길이 L 이 곧 가스수요의 대리변수**이기 때문이다. 수요가 많으면 버너가 자주 켜져
#     OFF 런이 짧고, 동시에 수조에서 빼가는 열이 크다. 즉 L 과 m_gas 가 강하게 음의 상관이다.
#     런마다 m_gas 가 다른데 전역 상수 하나로 놓았으니 모형 자체가 오설정이었다.
#     **OFF 구간 열수지로는 m_gas 를 식별할 수 없다.** (사용자가 지적한 표본 선택 효과의 실체)


# ---------------------------------------------------------------------------
# 채택: 무부하 냉각 직접 측정
# ---------------------------------------------------------------------------
# 위 문제를 피하는 유일한 길은 회귀를 버리고 **m_gas·c_p·ΔT_gas 항이 0 인 구간만 고르는 것**이다.
#     버너 OFF ∧ 통가스 없음(ΔT_gas ≈ 0)  →  C_eff·|dT_bath/dt| = Q_loss
# 미지수가 하나뿐이라 회귀도, 절편 가정도 필요 없다. 런별로 값을 내고 중앙값을 쓴다.
#
# ⚠ A·B 는 이 방법이 통하지 않는다. ΔT_gas < 1℃ 인 런이 15년간 6개뿐이다
#   (A/B 는 상시 통가스 계열이라 무부하 상태가 사실상 없다). O·P 만 측정 가능하다.

NOFLOW_DT_MAX = 1.0     # 무통가스 판정 [℃] — 런 전체의 ΔT_gas 최대값 기준
NOFLOW_MIN_LEN = 60     # 최소 런 길이 [분]
NOFLOW_MIN_RUNS = 20


def no_load_cooling(tr: pd.DataFrame, burner: pd.Series, u: str) -> pd.DataFrame:
    """무부하(버너 OFF + 통가스 없음) 냉각 런. 반환: L[분], slope[K/s], gmax[℃]."""
    bath, gas = tr[f"TI-D2{u}"], tr[f"dT_{u}"]
    ok = (burner == 0) & bath.notna() & (bath > MIN_BATH_C) & gas.notna()
    run = ((ok != ok.shift()) | (tr["segment_id"] != tr["segment_id"].shift())).cumsum()

    f = pd.DataFrame({"run": run[ok], "T": bath[ok].to_numpy(), "g": gas[ok].to_numpy()},
                     index=tr.index[ok])
    f = f[f.groupby("run").cumcount() >= SETTLE_MIN]
    f["t"] = f.groupby("run").cumcount().to_numpy() * 60.0

    g = f.groupby("run")
    n, tb, Tb = g.size(), g["t"].mean(), g["T"].mean()
    stt = g["t"].apply(lambda s: float(((s - s.mean()) ** 2).sum()))
    stT = f.assign(tT=f["t"] * f["T"]).groupby("run")["tT"].sum() - n * tb * Tb
    out = pd.DataFrame({"L": n, "slope": stT / stt.replace(0.0, np.nan),
                        "gmax": g["g"].max(), "T_bath": Tb, "t0": g.apply(lambda d: d.index[0])})
    return out[(out["L"] >= NOFLOW_MIN_LEN) & (out["gmax"] < NOFLOW_DT_MAX)
               & out["slope"].notna()]


def q_loss_kw(runs: pd.DataFrame, c_eff: float) -> dict:
    """무부하 런들에서 Q_loss [kW]. 회귀 없음 — 런별 값의 중앙값과 사분위."""
    if len(runs) < NOFLOW_MIN_RUNS or not np.isfinite(c_eff):
        return {"q_loss_kw": np.nan, "q25": np.nan, "q75": np.nan,
                "n": len(runs), "T_bath": np.nan}
    q = (-c_eff * runs["slope"] / 1e3).to_numpy(float)
    return {"q_loss_kw": float(np.median(q)), "q25": float(np.percentile(q, 25)),
            "q75": float(np.percentile(q, 75)), "n": len(runs),
            "T_bath": float(runs["T_bath"].median())}
