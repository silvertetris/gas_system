"""PINN 파일럿 설정 — 표본으로 먼저 확인한다.

## 표본 선택 근거

0℃ 미만 사건이 **2016년 이후 사실상 사라졌다**(M계열 2011~2014 11~16% → 2018~2026 0.0~0.2%,
docs/08 §2). 최근 구간으로 파일럿을 하면 사건이 거의 없어 **검증 자체가 불가능**하다.
→ **사건이 많은 2012~2016** 을 쓴다. 5년, 1시간 격자 ≈ 43,800행.

전체(15년·1분·780만행)로 바로 가지 않는 이유는 lstm/autoencoder 파일럿과 같다 —
**되는지 먼저 확인**하고, 물리 정합·GBM 대비를 통과한 뒤 전 구간으로 확장한다.

## 물리 연쇄 (docs/08, §15)

    주배관 PI21X → 필터 → 히터+바이패스 → 정압기 → 수요처 TI61x
    (P1) 줄-톰슨   T_61 = T_hdr − μ_JT·ΔP − Q_pipe      μ_JT 실측 0.34~0.41 ℃/bar
    (P2) 합류점    T_hdr = (1−β)·T_out_mix + β·T_in      §15 검증 r=0.87/0.84
"""
from __future__ import annotations

from pathlib import Path

from prex import config as p_config
from prex_multi import config as m_config

PROJECT_ROOT = p_config.PROJECT_ROOT
TREND_PARQUET = m_config.OUTPUT_DIR / "trend_all.parquet"
RAW_TREND_DIR = p_config.TREND_DIR
OUTPUT_DIR = PROJECT_ROOT / "pinn" / "test" / "output"

# --- 표본 구간 (사건이 많은 시기)
SAMPLE_START = "2012-01-01"
SAMPLE_END = "2017-01-01"
TRAIN_END = "2015-01-01"     # 3년
VALID_END = "2016-01-01"     # 1년 — 보정·조기종료용
# 시험: 2016-01 ~ 2017-01 (1년, M 발생률 4.5% · Z 2.3%)

FREQ = "1h"
HORIZON_H = 6
PURGE_H = 24                 # 분할 경계에서 버릴 시간 (+ 지평)

FREEZE_C = 0.0               # 물리적 빙결점
MPA_TO_BAR = 10.0            # `PI21X`·`PI61x` 는 MPa. μ_JT 가 ℃/bar 이므로 변환 필요

TRAINS = {
    "M": {"t61": "TI61M", "p61": "PI61M", "hdr": "TSL_TSH_TSHH-31M",
          "t_in": "TI21Y", "units": ["A", "B"]},
    "Z": {"t61": "TI61Z", "p61": "PI61Z", "hdr": "TSL_TSH_TSHH-31Z",
          "t_in": "TI21Z", "units": ["O", "P"]},
}
P_IN = "PI21X"

# --- 물리 모수 사전(prior). 학습하되 이 범위를 벗어나면 기각한다.
MU_JT_INIT = 0.40            # ℃/bar — 천연가스 문헌값 0.4~0.5, 실측 0.34~0.41
MU_JT_MIN, MU_JT_MAX = 0.25, 0.60
LOSS_INIT_C = 1.0            # 배관 열손실 상당 온도강하 [℃] ≥ 0

# --- 학습
EPOCHS = 2000
LR = 1e-3                    # 3e-3 은 검증손실이 진동했다(물리항 추가 후)
HIDDEN = 64
BATCH = 1024
N_MC = 64                    # 몬테카를로 표본 (비선형 물리라 분산 전파에 필요)
W_PHYS = 1.0                 # 물리 잔차 가중 (동시점 관측 (hdr, dp, t61) 정합)
W_STATE = 2.0                # 중간 상태 지도 가중
                             # ⚠ 이게 0 이면 분해가 무의미해진다 — model.py 의 1차 버그 참고
                             # ⚠ 0.5 는 **너무 작았다**(4차 버그). 물리층에서 `T_g = a + b·T_in`
                             #    이라 신경망이 `T_in` 예측을 왜곡해 `T_61` 을 맞추는 쪽이 이득이
                             #    된다 — 입구온도 RMSE 23℃, 예측 AUC 0.94→0.60 으로 붕괴했다.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 히터 물리 (2026-09-12 신설) — (Q1)~(Q4) 를 물리층에 넣기 위한 상수
# ---------------------------------------------------------------------------
# 이전 파일럿은 `T_hdr` 을 신경망이 그냥 예측했다 — **히터 물리가 통째로 없었다.**
# `m·c_p·ΔT = U·A·ΔT_lm` 를 넣는다. 수조온도가 관 길이방향으로 일정한 단일유체
# 열교환기라 이 식은 ε-NTU 와 **대수적으로 동일**하다(실측 870만행 검증, 상대오차 1e-15):
#
#     (T_out − T_in)/ΔT_lm = −ln(1−ε) = NTU = U·A/(m·c_p)
#
# 그래서 LMTD 형태를 따로 쓰지 않고 ε-NTU 로 구현한다 — 같은 식이다.
#
# ⚠ 제작도서 상수를 **고정**하면 역산이 검증에 실패한다(문서 07 §2-1). 그래서 여기 값들은
#    **초기값·기준값**이고, `U·A` 보정계수와 유효화력률은 **학습 모수**로 둔다.
CP_GAS_J_KGK = p_config.HTR_CP_GAS_J_KGK          # 2739.8 J/kg·K
ALPHA_IO_KCAL = p_config.HTR_ALPHA_IO_KCAL        # 2048 — 가스측(내막), ∝ m^0.8
ALPHA_O_KCAL = p_config.HTR_ALPHA_O_KCAL          # 863.1 — 수조측(외막), 유량 무관
GAS_SIDE_EXPONENT = p_config.GAS_SIDE_EXPONENT    # 0.8 (Dittus-Boelter)
# 제작도서 U=513.6 에는 오염저항이 이미 포함 → 청정저항과의 차가 R_f
R_FOUL = 1.0 / p_config.HTR_U_KCAL - (1.0 / ALPHA_IO_KCAL + 1.0 / ALPHA_O_KCAL)
Q_BURNER_DESIGN_W = p_config.HTR_QBURNER_DESIGN_W  # ≈ 0.996 MW (연료소비 90 Nm³/h × LHV)

UNIT_DESIGN = {u: {"m_d": m_config.design_mgas_kg_s(u),
                   "ua_d": m_config.scaled_ua_w_per_k(u)} for u in m_config.UNITS}

# ⚠ 아래는 **더 이상 쓰지 않는다**(문서 10). 기록을 위해 남긴다.
#   · `UA_SCALE_LOG_MAX`·`FIRE_*`·`CALORIMETRY`: ε-NTU 유량 역산과 (Q1) 수조 열수지용이었다.
#     역산은 부호가 반대였고((Q4) 홀드아웃 R² −2.71), (Q1) 은 버너 발열이 ε 역산 흡수열의
#     1/3~1/4 뿐이라 기각됐다. 유량 계측이 생기면 되살릴 값들이다.
CALORIMETRY = {
    "A": {"c_eff_mjk": 38.0, "q_loss_kw": 5.3, "dt_ref_c": 33.7},
    "B": {"c_eff_mjk": 33.8, "q_loss_kw": 3.6, "dt_ref_c": 27.2},
    "O": {"c_eff_mjk": 300.9, "q_loss_kw": 35.0, "dt_ref_c": 32.6},
    "P": {"c_eff_mjk": 82.3, "q_loss_kw": 22.3, "dt_ref_c": 45.0},
}
UA_SCALE_LOG_MAX = 2.0
FIRE_MIN, FIRE_MAX = 0.05, 1.0

# ---------------------------------------------------------------------------
# 히터 블록 상수 (2026-09-13 개정 — 문서 10 반영)
# ---------------------------------------------------------------------------
# 이전 판은 신경망이 유량비 `x_u` 를 내고 `ε` 을 ε-NTU 로 **역산**했다. 그 방향이 틀렸다.
# 1분 원해상도 전기간·대조군 12개 홀드아웃 검정 (문서 10 §3):
#     w_u ∝ m_d,u·ε_u (채택)   M R² +0.522 / Z +0.327
#     영가설 (ε 시간치환)       M −0.018 / Z −0.049
#     ε-NTU 역산 (구 판)       M −2.712 / Z −1.907     ← 영가설보다도 나쁘다
# 밸브 최소개도에서 혼합비를 직접 풀면 Spearman(w_A, 역산 w_A) = −0.735/−0.547 (대조군 ±0.002).
# 원인: 정지 히터의 출구 열전대가 정체 가스를 읽어 ε→0 → 역산이 "대유량"으로 오독.
# ⇒ `ε` 을 **관측값 그대로** 쓴다. `U·A` 와 `m` 의 분리는 포기한다(문서 07 §2 유지).
CP_GAS_J_KGK = p_config.HTR_CP_GAS_J_KGK          # 2739.8 J/kg·K — 혼합 가중 규모에만 쓴다
UNIT_DESIGN = {u: {"m_d": m_config.design_mgas_kg_s(u)} for u in m_config.UNITS}

EPS_LO, EPS_HI = 0.02, 0.99   # 물리층에 넣기 전 ε 표본의 유계화
MIN_DRIVE_C = 5.0             # ε 이 정의되는 구동온도차 하한 (T_bath − T_in)
DRIVE_SOFT_C = 2.0            # 그 임계의 부드러움. 미분 가능해야 한다
BETA_KNOTS = 8                # β(밸브) 단조맵 매듭 수 (밸브개도 분위)

# --- 잔차 가중. 전부 **동시점 실측**에 대한 잔차다(미래값 아님).
# ⚠ (Q3) 잔차는 **항등적으로 0** 이다 — ε 을 실측으로 쓰면 T_out = T_in + ε·(T_bath − T_in)
#   이 ε 의 정의와 같다. 그래서 손실에 넣지 않는다.
W_Q4 = 2.0                    # 합류점 헤더온도 T_hdr — (Q4) 동시점 정합
                              # (Q5)·(Q6) 공급온도 정합은 W_PHYS 를 쓴다.
# ⚠ **잔차 정규화 필수**(4차 버그). 물리 잔차는 ℃² 로 10~100, NLL 은 O(1) 이다. 그냥 더하면
#   물리항이 손실을 지배해 상태 헤드가 학습되지 않는다(입구온도 RMSE 23℃, AUC 0.94→0.60).
#   각 잔차를 해당 관측량의 훈련구간 분산으로 나눠 무차원 O(1) 로 맞춘다.
PATIENCE = 200                # 조기종료 인내. 100 에폭에서 아직 손실이 30배 줄던 중이었다

# --- 기록용. 유량 계측이 생기면 되살릴 값들이다 (지금은 **쓰지 않는다**).
#   (Q1) 수조 열수지는 기각됐다: 통가스 시간대 버너 duty 평균이 A 0.036·B 0.084·O 0.145
#   뿐인데(=36·84·144 kW) ε 역산 흡수열은 127·124·644 kW 다. 3~4배 부족하고 수조
#   저장항(−5~−20 kW)으로도 못 메운다. 문서 09 §4.
CALORIMETRY = {
    "A": {"c_eff_mjk": 38.0, "q_loss_kw": 5.3, "dt_ref_c": 33.7},
    "B": {"c_eff_mjk": 33.8, "q_loss_kw": 3.6, "dt_ref_c": 27.2},
    "O": {"c_eff_mjk": 300.9, "q_loss_kw": 35.0, "dt_ref_c": 32.6},
    "P": {"c_eff_mjk": 82.3, "q_loss_kw": 22.3, "dt_ref_c": 45.0},
}
Q_BURNER_DESIGN_W = p_config.HTR_QBURNER_DESIGN_W  # ≈ 0.996 MW
