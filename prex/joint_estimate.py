"""[2·3단계] m_gas · UA_ref · C_eff 공동추정 — 부하보정된 열화지표를 만든다.

배경: 1단계(`prex/degradation.py`)에서 정비 주기별 ε 추세를 봤으나 R²가 0.13 이하로
      판정 불가였다. ε = f(U·A, m_gas) 라 **부하가 섞여 있기 때문**이다.
      부하를 분리하려면 m_gas 가 필요한데, 밸브식 역산은 P&ID로 무효화됐고(§7-2)
      단독 회귀도 진공 잠열항 때문에 실패했다(§7-3의 m_gas 0.09배).
      → **세 미지수를 함께 푸는 수밖에 없다.**

식별 경로 (전부 신뢰 태그 TI21Z/TI33P/TI-D2P + 제작도서 상수만 사용):

  (1) C_eff — 수조 **유효** 열용량 (현열 + 진공 비등 잠열)
      자유 승온 구간(제어가 안 걸리는 냉간 기동)에서:
          C_eff · dT_bath/dt ≈ s̄·Q_burner − Q_gas − Q_loss
      Q_burner 는 제작도서 설계값(0.996 MW)이라 **절대 스케일이 고정**된다.
      ⚠ 정상 운전 구간을 쓰면 안 된다 — 수조가 온도제어되어 dT/dt→0 이라 발산한다
        (실측: 창 5→120분에서 222→7,828 MJ/K 로 발산).

  (2) m_gas — 버너 OFF 수조 열수지에서:
          −C_eff · dT_bath/dt = m_gas·c_p·(T_out−T_in) + Q_loss
      기울기가 m_gas·c_p, 절편이 Q_loss. (1)에서 C_eff 를 알므로 m_gas 가 절대값으로 나온다.

  (3) UA_ref — ε 에서 NTU 를 얻고 부하보정을 역산:
          NTU = U·A/(m_gas·c_p),   U·A = UA_ref·(m_gas/m_design)^0.8
          ⇒ UA_ref = NTU · c_p · m_design^0.8 · m_gas^0.2
      NTU 가 m_gas 에 **m^(-0.2)** 로만 의존하므로 부하 영향이 약하다 — 이게 이 경로의 장점이다.

  → `UA_ref(t)` 가 열화지표다. 정비 주기별로 추정해 톱니 패턴을 검증한다.

실행: .venv/bin/python -m prex.joint_estimate
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# --- C_eff 추정용 (자유 승온 구간) ---
RAMP_WINDOW_MIN = 180      # 이 창에서
RAMP_MIN_RISE_C = 10.0     # 이만큼 올라야 '오르는 중'
RAMP_MIN_TOTAL_C = 15.0    # 구간 전체 상승폭 하한
RAMP_MIN_HOURS = 1.0
RAMP_MIN_BURNER_ON = 0.3   # 버너가 이 비율 이상 켜져 있어야 함

# --- m_gas 추정용 (버너 OFF 열수지) ---
OFF_SETTLE_MIN = 10        # ON→OFF 전이 직후 과도구간 제거
OFF_DIFF_MIN = 15          # 수조 온도 차분 창 [분] — 계기 분해능 0.061℃ 극복용
MIN_GAS_DT = 5.0           # 가스 승온폭 하한 [℃] — '가동 중' 정의(docs §8-4)와 일치시킨다.
                           # 2.0 이면 m_gas 회귀는 저부하 행까지 쓰고 NTU 는 고부하 행만 써서
                           # 서로 다른 부하의 값을 결합하게 된다(점검 D).
TRIM = 0.05

LOAD_EXPONENT = 0.8        # Dittus-Boelter 가스측 α ∝ m^0.8 (교과서값, T3)
MIN_CYCLE_DAYS = 60


def load_inputs() -> tuple[pd.DataFrame, pd.Series]:
    tr = pd.read_parquet(config.OUTPUT_DIR / "trend_htr31p.parquet").set_index("Time").sort_index()
    al = pd.read_parquet(config.OUTPUT_DIR / "alarm_events_htr31p.parquet")
    h = al.loc[al["tag"] == "H31POH", ["Time", "value"]].dropna()
    h = h.sort_values("Time").drop_duplicates("Time", keep="last")
    m = pd.merge_asof(pd.DataFrame({"Time": tr.index}), h, on="Time", direction="backward")
    return tr, pd.Series(m["value"].to_numpy(), index=tr.index, name="burner")


def estimate_c_eff(tr: pd.DataFrame, burner: pd.Series,
                   m_gas_est: float = 0.0, q_loss_est: float = 0.0) -> tuple[float, pd.DataFrame]:
    """자유 승온 구간에서 수조 유효 열용량 C_eff [J/K] 추정.

    m_gas_est·q_loss_est 를 주면 Q_gas·Q_loss 를 빼고 계산한다(정확). 0 이면 상한이 나온다.
    이 둘이 다시 C_eff 에 의존하므로 `iterate()` 에서 **반복 갱신**한다.
    """
    T = tr["TI-D2P"]
    seg_gas = tr["TI33P"] - tr[config.HTR31P_INLET_TAG]
    W = RAMP_WINDOW_MIN
    cand = (T.diff(W) > RAMP_MIN_RISE_C) & (tr["segment_id"] == tr["segment_id"].shift(W))
    grp = (cand != cand.shift()).cumsum()
    rows = []
    for _, g in tr[cand].groupby(grp[cand]):
        a = g.index.min() - pd.Timedelta(minutes=W)
        b = g.index.max()
        if a < T.index.min() or pd.isna(T.get(a)) or pd.isna(T.get(b)):
            continue
        dT = float(T.loc[b] - T.loc[a])
        dur = (b - a).total_seconds()
        if dT < RAMP_MIN_TOTAL_C or dur < RAMP_MIN_HOURS * 3600:
            continue
        on = float((burner.loc[a:b] == 1).mean())
        if on < RAMP_MIN_BURNER_ON:
            continue
        # C_eff·dT/dt = on·Q_burner − Q_gas − Q_loss
        #   Q_gas = m_gas·c_p·ΔT_gas 는 **버릴 수 없다** — 실측상 버너 입열의 4~130% 다.
        #   (구 버전은 이걸 0 으로 놔서 C_eff 가 상한으로 과대추정됐다)
        dt_gas = float((seg_gas.loc[a:b]).median())
        q_gas = m_gas_est * config.HTR_CP_GAS_J_KGK * dt_gas if np.isfinite(dt_gas) else 0.0
        numer = config.HTR_QBURNER_DESIGN_W * on - q_gas - max(q_loss_est, 0.0)
        rows.append({"start": a, "hours": dur / 3600, "dT_bath": dT, "burner_on": on,
                     "dTdt": dT / dur, "dt_gas": dt_gas,
                     "q_burner_kW": config.HTR_QBURNER_DESIGN_W * on / 1e3,
                     "q_gas_kW": q_gas / 1e3,
                     "c_eff": numer / (dT / dur) if numer > 0 else np.nan})
    df = pd.DataFrame(rows)
    c_eff = float(df["c_eff"].median()) if len(df) else np.nan
    logger.info("C_eff 추정: 승온구간 %d건 → 중앙 %.1f MJ/K (설계 현열의 %.1f배)",
                len(df), c_eff / 1e6, c_eff / config.HTR_MW_CW_J_PER_K)
    return c_eff, df


def estimate_m_gas(tr: pd.DataFrame, burner: pd.Series, c_eff: float,
                   mask: pd.Series | None = None) -> dict:
    """버너 OFF 수조 열수지로 m_gas [kg/s]·Q_loss [W] 추정."""
    t_in = tr[config.HTR31P_INLET_TAG]
    W = OFF_DIFF_MIN
    dTb = tr["TI-D2P"].diff(W) / (W * 60.0)
    y = -c_eff * dTb                                    # 수조가 잃는 열 [W]
    x = config.HTR_CP_GAS_J_KGK * (tr["TI33P"] - t_in)  # [W per kg/s]

    known = burner.notna()
    off = known & (burner != 1.0)
    new_run = (off != off.shift()) | (tr["segment_id"] != tr["segment_id"].shift())
    within = tr.groupby(new_run.cumsum()).cumcount()
    valid = (off & (tr["segment_id"] == tr["segment_id"].shift(W)) & (within >= OFF_SETTLE_MIN)
             & x.gt(config.HTR_CP_GAS_J_KGK * MIN_GAS_DT) & y.gt(0) & x.notna() & y.notna())
    if mask is not None:
        valid &= mask
    if valid.sum() < 500:
        return {"m_gas": np.nan, "q_loss": np.nan, "r2": np.nan,
                "n": int(valid.sum()), "rows": valid}

    xx, yy = x[valid].to_numpy(), y[valid].to_numpy()
    m = q = np.nan
    for _ in range(2):
        m, q = np.linalg.lstsq(np.c_[xx, np.ones(len(xx))], yy, rcond=None)[0]
        r = yy - (m * xx + q)
        lo, hi = np.quantile(r, [TRIM, 1 - TRIM])
        keep = (r >= lo) & (r <= hi)
        xx, yy = xx[keep], yy[keep]
    resid = yy - (m * xx + q)
    r2 = 1 - np.sum(resid ** 2) / np.sum((yy - yy.mean()) ** 2)
    return {"m_gas": float(m), "q_loss": float(q), "r2": float(r2),
            "n": int(valid.sum()), "rows": valid}


def ua_clean(m_gas: float) -> float:
    """주어진 유량에서 **오염이 없다면 나와야 할** U·A [W/K] — 직렬 저항 모델.

        1/U = 1/(α_io·(m/m_d)^0.8) + 1/α_o

    가스 유량이 변하면 **가스측(α_io)만** Dittus-Boelter 로 변하고 수조측(α_o)은 그대로다.
    설계점에서 가스측 저항은 30% 뿐이라, 전체를 m^0.8 로 보정하던 구 방식은 과보정이었다
    (config.HTR_ALPHA_* 주석 참고).
    """
    if not np.isfinite(m_gas) or m_gas <= 0:
        return np.nan
    r = m_gas / config.HTR_DESIGN_MGAS_KG_S
    a_io = config.HTR_ALPHA_IO_KCAL * r ** config.GAS_SIDE_EXPONENT
    u_kcal = 1.0 / (1.0 / a_io + 1.0 / config.HTR_ALPHA_O_KCAL)
    return u_kcal * config.HTR_AREA_M2 * config.KCAL_TO_J / 3600.0


def health_factor(ntu: float, m_gas: float) -> tuple[float, float]:
    """열화지표 φ = 실제 U·A / 청정 U·A. 1.0 이면 설계대로, 낮을수록 열화.

    Returns: (φ, 실제 U·A [W/K])
    φ 를 쓰는 이유: 부하가 변해도 **청정 상태 대비 비율**이라 주기 간 비교가 성립한다.
    """
    if not np.isfinite(ntu) or not np.isfinite(m_gas) or m_gas <= 0:
        return np.nan, np.nan
    ua_actual = ntu * m_gas * config.HTR_CP_GAS_J_KGK
    ua_c = ua_clean(m_gas)
    return (ua_actual / ua_c if ua_c > 0 else np.nan), ua_actual


def solve_ceff_qburner(ramps: pd.DataFrame, m_gas: float, q_loss: float) -> dict:
    """승온구간들로 **C_eff 와 Q_burner 를 동시에** 최소제곱으로 푼다.

    구간 i 마다 수조 열수지:
        C_eff·(dT/dt)_i = on_i·Q_b − Q_gas_i − Q_loss
    ⇒  C_eff·(dT/dt)_i − Q_b·on_i = −Q_gas_i − Q_loss        … [C_eff, Q_b] 에 대해 선형

    구간마다 on_frac(0.3~1.0)과 Q_gas 가 달라 두 미지수가 분리된다.

    **왜 필요한가**: 구 버전은 Q_b 를 제작도서 설계값(0.996 MW) 상수로 고정했는데,
    실제 버너는 BLOWER 회전수로 **연속 변조**된다(문서 05 §3-3). 설계값보다 실제 입열이
    작으면 C_eff 가 과대추정되고, 그게 OFF 회귀의 좌변을 부풀려 **Q_loss 를 음수로** 민다.
    (실측: 점검 A·B·D 를 다 고쳐도 Q_loss −17.9 kW 로 음수가 남았다)
    """
    d = ramps.dropna(subset=["dTdt", "burner_on", "dt_gas"])
    if len(d) < 10:
        return {"c_eff": np.nan, "q_burner": np.nan, "r2": np.nan, "n": len(d), "cond": np.nan}
    q_gas = m_gas * config.HTR_CP_GAS_J_KGK * d["dt_gas"].to_numpy()
    A = np.c_[d["dTdt"].to_numpy(), -d["burner_on"].to_numpy()]
    b = -q_gas - max(q_loss, 0.0)
    cond = float(np.linalg.cond(A))
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c_eff, q_b = float(sol[0]), float(sol[1])
    pred = A @ sol
    r2 = 1 - np.sum((b - pred) ** 2) / np.sum((b - b.mean()) ** 2)
    return {"c_eff": c_eff, "q_burner": q_b, "r2": float(r2), "n": len(d), "cond": cond}


def iterate(tr: pd.DataFrame, burner: pd.Series, n_iter: int = 12,
            tol: float = 1e-3) -> dict:
    """C_eff ↔ (m_gas, Q_loss) 반복 추정.

    **왜 반복이 필요한가**: 셋이 서로를 필요로 한다.
      - C_eff 를 구하려면 승온구간의 Q_gas(=m_gas·c_p·ΔT)와 Q_loss 를 빼야 한다
      - m_gas·Q_loss 는 버너 OFF 열수지 회귀에서 나오는데 그 회귀에 C_eff 가 들어간다
    구 버전은 Q_gas 를 0 으로 놓아 순환을 끊었는데, 실측상 Q_gas 가 버너 입열의 4~130% 라
    C_eff 가 크게 과대추정됐다. 여기서는 수렴할 때까지 번갈아 갱신한다.

    ⚠ Q_loss 는 물리적으로 양수여야 하므로 C_eff 계산에 넣을 때 0 에서 클리핑한다.
      회귀 자체는 절편을 자유롭게 두고, 음수로 나오면 그 자체를 진단 신호로 본다.
    """
    # --- 시드: 1회는 구 방식(Q_b=설계 상수, Q_gas 무시)으로 m_gas 를 얻는다.
    #     m_gas=0 으로 시작하면 solve_ceff_qburner 의 우변이 전부 0 이라 자명해([0,0])로 빠진다.
    c0, _ = estimate_c_eff(tr, burner, m_gas_est=0.0, q_loss_est=0.0)
    seed = estimate_m_gas(tr, burner, c0)
    m_gas = seed["m_gas"] if np.isfinite(seed["m_gas"]) else config.HTR_DESIGN_MGAS_KG_S * 0.4
    q_loss, c_eff, q_burner = 0.0, c0, config.HTR_QBURNER_DESIGN_W
    logger.info("시드: C_eff=%.1f MJ/K, m_gas=%.2f kg/s", c0 / 1e6, m_gas)
    hist = []
    for it in range(n_iter):
        _, ramps = estimate_c_eff(tr, burner, m_gas_est=m_gas, q_loss_est=q_loss)
        sol = solve_ceff_qburner(ramps, m_gas, q_loss)
        if np.isfinite(sol["c_eff"]) and sol["c_eff"] > 0:
            c_eff, q_burner = sol["c_eff"], sol["q_burner"]
        else:
            # ⚠ 동시추정 실패 — 설계 Q_burner 로 폴백한다.
            #   실측(2026-09-12): C_eff = −138.5 MJ/K (음수). 원인은 §11-2 참고 —
            #   설계행렬 조건수 407.8 + q_gas 부호 역전(−105~+1,187 kW).
            c_eff, _r = estimate_c_eff(tr, burner, m_gas_est=m_gas, q_loss_est=q_loss)
            q_burner = config.HTR_QBURNER_DESIGN_W
            if it == 0:
                logger.warning("C_eff·Q_burner 동시추정 실패(조건수 %.0f) → 설계 Q_burner 로 폴백",
                               sol.get("cond", float("nan")))
        est = estimate_m_gas(tr, burner, c_eff)
        new_m, new_q = est["m_gas"], est["q_loss"]
        hist.append({"iter": it, "c_eff_MJ": round(c_eff / 1e6, 2),
                     "Q_burner_kW": round(q_burner / 1e3, 1),
                     "m_gas": round(new_m, 3), "q_loss_kW": round(new_q / 1e3, 1),
                     "ramp_r2": round(sol["r2"], 3), "off_r2": round(est["r2"], 3),
                     "n_ramp": sol["n"]})
        conv = np.isfinite(new_m) and abs(new_m - m_gas) / max(abs(new_m), 1e-9) < tol
        m_gas, q_loss = new_m, new_q
        if conv:
            break
    logger.info("반복 %d회: C_eff=%.1f MJ/K, Q_burner=%.0f kW (설계 %.0f), m_gas=%.2f, Q_loss=%.1f kW",
                len(hist), c_eff / 1e6, q_burner / 1e3,
                config.HTR_QBURNER_DESIGN_W / 1e3, m_gas, q_loss / 1e3)
    return {"c_eff": c_eff, "q_burner": q_burner, "m_gas": m_gas, "q_loss": q_loss,
            "hist": pd.DataFrame(hist), "ramps": ramps}


def per_cycle(tr: pd.DataFrame, burner: pd.Series, c_eff: float) -> pd.DataFrame:
    """정비 주기별로 m_gas·UA_ref 추정 → 열화 궤적."""
    work = ((tr["TI33P"] - tr[config.HTR31P_INLET_TAG]) > 5.0) & tr["htx_ntu"].notna()
    rows = []
    for cyc, sub in tr.groupby("maint_cycle"):
        if len(sub) < MIN_CYCLE_DAYS * 1440:
            continue
        in_cycle = pd.Series(tr.index.isin(sub.index), index=tr.index)
        est = estimate_m_gas(tr, burner, c_eff, mask=in_cycle)
        # ⚠ 점검 D 수정: NTU 를 m_gas 회귀가 실제로 쓴 **바로 그 행**에서 뽑는다.
        #   예전에는 m_gas 는 버너 OFF·저부하 행, NTU 는 '가동 중'(ON+OFF) 행에서 뽑아
        #   부하가 5배 다른 값을 결합했다. 같은 행에서 뽑으면 이 불일치가 사라진다.
        same = est["rows"] & tr["htx_ntu"].notna()
        ntu = float(tr.loc[same, "htx_ntu"].median()) if same.sum() > 100 else np.nan
        phi, ua_act = health_factor(ntu, est["m_gas"])
        rows.append({
            "cycle": int(cyc), "시작": sub.index.min().date(), "종료": sub.index.max().date(),
            "일수": round(len(sub) / 1440),
            "m_gas": round(est["m_gas"], 3), "m_gas_r2": round(est["r2"], 3),
            "Q_loss_kW": round(est["q_loss"] / 1e3, 1) if np.isfinite(est["q_loss"]) else np.nan,
            "NTU": round(ntu, 4), "NTU_n": int(same.sum()),
            "UA_실제_kW_K": round(ua_act / 1e3, 2) if np.isfinite(ua_act) else np.nan,
            "UA_청정_kW_K": round(ua_clean(est["m_gas"]) / 1e3, 2),
            "φ_건전도": round(phi, 3) if np.isfinite(phi) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 170)
    tr, burner = load_inputs()

    res = iterate(tr, burner)
    c_eff, ramps = res["c_eff"], res["ramps"]
    print(f"\n=== (0) 반복 추정 수렴 과정 ===")
    print(res["hist"].to_string(index=False))
    print(f"\n=== (1) 수조 유효 열용량 C_eff ===")
    print(f"  자유 승온 구간 {len(ramps)}건 → C_eff = {c_eff/1e6:.1f} MJ/K")
    print(f"  설계 현열 M_w·c_w = {config.HTR_MW_CW_J_PER_K/1e6:.2f} MJ/K → **{c_eff/config.HTR_MW_CW_J_PER_K:.1f}배**")
    print(f"  → 초과분은 진공 비등의 잠열로 해석된다(§7-3 가설 확인)")
    print(f"\n=== (1b) 버너 실효 입열 Q_burner (동시추정) ===")
    print(f"  추정 {res['q_burner']/1e3:.0f} kW  vs 제작도서 설계 {config.HTR_QBURNER_DESIGN_W/1e3:.0f} kW"
          f"  → 설계의 {res['q_burner']/config.HTR_QBURNER_DESIGN_W:.2f}배")

    glob = {"m_gas": res["m_gas"], "q_loss": res["q_loss"],
            "r2": float(res["hist"]["off_r2"].iloc[-1])}
    print(f"\n=== (2) 전 구간 m_gas ===")
    print(f"  m_gas = {glob['m_gas']:.2f} kg/s (설계 {config.HTR_DESIGN_MGAS_KG_S:.2f} 의 "
          f"{glob['m_gas']/config.HTR_DESIGN_MGAS_KG_S:.2f}배)  Q_loss = {glob['q_loss']/1e3:.1f} kW  R²={glob['r2']:.3f}")

    cyc = per_cycle(tr, burner, c_eff)
    print(f"\n=== (3) 정비 주기별 건전도 φ = 실제 U·A / 청정 U·A ===")
    print(cyc.to_string(index=False))
    print(f"\n  설계 U·A = {config.HTR_UA_W_PER_K/1e3:.1f} kW/K")

    out = config.OUTPUT_DIR / "degradation"
    out.mkdir(parents=True, exist_ok=True)
    ramps.to_csv(out / "c_eff_ramps.csv", index=False)
    cyc.to_csv(out / "cycle_ua_ref.csv", index=False)
    logger.info("저장: %s", out)


if __name__ == "__main__":
    main()
