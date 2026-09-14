"""히터 열전달식 검정 설정 — 예측이 아니라 **식이 성립하는지**만 본다.

## 무엇을 검정하나

    (Q2) m_u·c_p·(T_out,u − T_in) = U_u·A_u·ΔT_lm,u
         ⇔  ε_u = 1 − exp(−U_u·A_u/(m_u·c_p))
    (Q3) T_out,u = T_in + ε_u·(T_bath,u − T_in)
    (Q4) T_hdr   = Σ_u f_u·T_out,u + β·T_in,     f_u = m_u/M, β = m_byp/M

`ε` 은 세 온도로 직접 계산되므로 **(Q3) 자체는 항등식**이다 — 검정 대상이 아니다.
검정 가능한 것은 두 가지다.

### 1. 무모수 국소 기울기 검정 (가장 강하다)

(Q3) 이 맞으면 유량이 고정된 국소 조건에서

    ∂T_out/∂T_bath = ε        ∂T_out/∂T_in = 1 − ε

이어야 한다. **자유 모수가 없다.** 조건(입구온도·밸브개도·버너duty)을 좁게 자른 구간에서
회귀 기울기를 재고 같은 구간의 ε 과 비교한다. 틀리면 식이 틀린 것이다.

### 2. 시불변 상수 제약 하의 합류점 예측

행마다 미지수 3개(x_A, x_B, x_byp)·관측 3개(T_out_A, T_out_B, T_hdr)면 자유도 0 이라
아무것도 검정하지 못한다. **U·A 는 시불변**, **β 는 밸브개도의 단조함수**로 묶으면
미지수가 693만 → 자유도가 행 수만큼 생긴다(과결정). 그 상태에서 `T_hdr` 예측을
**시간 분할 홀드아웃**으로 평가하고, 가중방식만 바꾼 **대조군**과 비교한다.

대조군이 핵심이다. 물리 가중이 균등·설계 가중보다 낫지 않으면 열전달식이
합류점을 설명한다고 말할 수 없다.
"""
from __future__ import annotations

from prex import config as p_config
from prex_multi import config as m_config

PROJECT_ROOT = p_config.PROJECT_ROOT
TREND_PARQUET = m_config.OUTPUT_DIR / "trend_all.parquet"
OUTPUT_DIR = PROJECT_ROOT / "pinn" / "physics" / "output"

VALID_FROM = m_config.HEADER_VALID_FROM       # 2013-06-01 — 헤더온도 태그 유효 시점
TRAIN_END = "2021-01-01"                      # 학습
TEST_FROM = "2022-01-01"                      # 홀드아웃 (1년 퍼지)

TRAINS = {
    "M": {"units": ["A", "B"], "t_in": "TI21Y", "hdr": "TSL_TSH_TSHH-31M", "valve": "ZI31D"},
    "Z": {"units": ["O", "P"], "t_in": "TI21Z", "hdr": "TSL_TSH_TSHH-31Z", "valve": "ZI31Q"},
}

# --- 열전달 상수 (제작도서). U·A 보정계수만 학습한다.
CP_GAS_J_KGK = p_config.HTR_CP_GAS_J_KGK
ALPHA_IO_KCAL = p_config.HTR_ALPHA_IO_KCAL
ALPHA_O_KCAL = p_config.HTR_ALPHA_O_KCAL
GAS_SIDE_EXPONENT = p_config.GAS_SIDE_EXPONENT
R_FOUL = 1.0 / p_config.HTR_U_KCAL - (1.0 / ALPHA_IO_KCAL + 1.0 / ALPHA_O_KCAL)
DESIGN_NTU = m_config.design_ntu("P")         # 0.4056 — 네 히터 공통 (설계유량비 스케일)
UNIT_MD = {u: m_config.design_mgas_kg_s(u) for u in m_config.UNITS}

# --- ε 유효 범위. 분모가 작으면 ε 이 잡음으로 발산한다.
MIN_DRIVE_C = 5.0                             # T_bath − T_in 하한
EPS_MIN, EPS_MAX = 0.02, 0.98                 # 밖이면 NTU 역산 분해능이 없다
X_MIN, X_MAX = 0.02, 15.0                     # 유량비 x = m/m_d 허용 범위

# --- β(밸브) 단조맵
BETA_KNOTS = 8                                # 밸브개도 분위 매듭 수

# --- 학습
EPOCHS = 60
LR = 3e-3
HIDDEN = 64
BATCH = 65_536
RANDOM_SEED = 42

# --- 국소 기울기 검정 구간 나누기
SLOPE_BINS = {"t_in": 8, "valve": 6, "duty": 3}
SLOPE_MIN_N = 2_000                           # 구간당 최소 행수
SLOPE_MIN_SPREAD_C = 3.0                      # 구간 내 T_bath 변동폭 하한 (기울기 식별용)
