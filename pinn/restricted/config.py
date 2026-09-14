"""제한 PINN — **데이터로 성립이 확인된 열전달식만** 쓴다 (2026-09-13).

## 무엇만 넣나

    (Q5) 줄-톰슨     T_reg = T_hdr − μ_JT·ΔP
    (Q6) 매설배관    T_61  = T_g + (T_reg − T_g)·exp(−N)
                     N     = U·A_배관/(m·c_p) + N₀           ← m 은 수조 열수지 유량(문서 12)
                     T_g   = a + b·T_in

| 식 | 근거 |
|---|---|
| (Q5) | μ_JT 학습 0.43~0.44, 문헌 0.4~0.5, KOGAS 공식 0.56 ℃/bar |
| (Q6) | 잔차구조 붕괴(β +0.68→−0.05), R² Z 0.56→0.78. **독립 유량으로 재확인**: Z 에서 exp(−N) 이 유량과 함께 0.37→0.72 로 증가, N ∝ 1/m 절편 +0.125 (거의 순수 1/m), U·A_배관 ≈ 10.2 kW/K |

## 무엇을 빼나 — 그리고 왜

| 식 | 제외 사유 |
|---|---|
| (Q2)(Q3) ε-NTU 히터 번들 | `TI33x` 로 만든 ε 이 유량과 **부호 반대**(문서 10·14) |
| (Q4) 합류점 혼합 | 가중치가 열전달식이 아니라 경험식이다 |
| (Q1) 수조 열수지 | 성립은 하나(문서 12) 빙결 경로 밖이다. 유량 `m` 을 만드는 데에만 쓴다 |

## N₀ 를 두는 이유

M 계열에서 N 대 1/m 적합의 절편이 **+1.374** 로 컸다 — 유량과 무관하게 헤더 신호가 사라지는
성분이 있다. Z 는 +0.125. 같은 형태를 두 계열에 쓰고 **데이터가 N₀ 를 정하게** 한다.
Z 에서 N₀ 가 0 근처로 학습되면 순수 매설배관 식이 성립한다는 뜻이다.

## "제한적" 의 의미

물리 잔차(동시점 T_61 정합)는 **열수지 유량이 실제로 관측된 시각에만** 건다. 유량이 없는
시각에 식을 강요하지 않는다. 평가도 두 부분집합(유량 관측 / 미관측)으로 나눠 보고한다.
"""
from __future__ import annotations

import os

from prex import config as p_config
from prex_multi import config as m_config

PROJECT_ROOT = p_config.PROJECT_ROOT
TREND_PARQUET = m_config.OUTPUT_DIR / "trend_all.parquet"
HEATFLOW_PARQUET = m_config.OUTPUT_DIR / "heatflow_cycles.parquet"
RAW_TREND_DIR = p_config.TREND_DIR
OUTPUT_DIR = PROJECT_ROOT / "pinn" / "restricted" / "output"
# --- 해상도 (2026-09-13). 환경변수 PINN_RES_MIN 으로 고른다. 기본 60분 = 이전 판과 완전히 같다.
#   코드 곳곳의 "행 1개 = 1시간" 가정을 시간 기준으로 바꿨다: 행 수 = 시간 × STEPS_PER_HOUR.
#   해상도마다 캐시·산출 폴더를 따로 둬 서로 덮어쓰지 않는다.
RES_MIN = int(os.environ.get("PINN_RES_MIN", "60"))
assert 60 % RES_MIN == 0, "RES_MIN 은 60 의 약수여야 한다"
STEPS_PER_HOUR = 60 // RES_MIN
RES_TAG = "" if RES_MIN == 60 else f"_{RES_MIN}min"
RAW_CACHE = OUTPUT_DIR / ("hourly_raw.parquet" if RES_MIN == 60 else f"raw{RES_TAG}.parquet")

# --- 기간 (문서 08 risk/ 와 같은 분할 → 직접 비교 가능)
VALID_FROM = "2013-10-01"     # 헤더온도 태그 첫 비영 (문서 14 §2)
TRAIN_END = "2019-01-01"
VALID_END = "2022-01-01"      # 시험 2022-01 ~
REGIME_FROM = None            # 문서 15 운전체제 변경 대응용. None = 전 기간

FREQ = "1h" if RES_MIN == 60 else f"{RES_MIN}min"
STEP_TD = __import__("pandas").Timedelta(minutes=RES_MIN)


def rows(hours: float) -> int:
    """시간 → 행 수."""
    return int(round(hours * STEPS_PER_HOUR))
HORIZON_H = 6                 # 24h 는 날씨가 결정한다(문서 08 §7-1)
PURGE_H = 24
FREEZE_C = 0.0
MPA_TO_BAR = 10.0

TRAINS = {
    "M": {"t61": "TI61M", "p61": "PI61M", "hdr": "TSL_TSH_TSHH-31M",
          "t_in": "TI21Y", "units": ["A", "B"]},
    "Z": {"t61": "TI61Z", "p61": "PI61Z", "hdr": "TSL_TSH_TSHH-31Z",
          "t_in": "TI21Z", "units": ["O", "P"]},
}
P_IN = "PI21X"
M_DESIGN_TOT = {t: sum(m_config.design_mgas_kg_s(u) for u in s["units"])
                for t, s in TRAINS.items()}

# --- 유량 (열수지 사이클 → 시간)
M_FFILL_H = 6                 # 사이클이 없는 시각은 과거방향으로만 채운다
BETA_MAX = 0.8                # 바이패스 보정 1/(1−β) 폭주 방지 (β→0.95 에서 106 kg/s 가 나왔다)
M_CAP_X_DESIGN = 2.0          # 계열 총유량 상한 = 설계 총유량 × 2
M_MIN = 0.05                  # kg/s. N = UA/(m·c_p) 발산 방지

# --- 물리 모수
CP_GAS_J_KGK = p_config.HTR_CP_GAS_J_KGK
MU_JT_INIT = 0.44
MU_JT_MIN, MU_JT_MAX = 0.25, 0.60
UA_PIPE_INIT_KW = {"M": 1.0, "Z": 10.0}   # 독립유량 검정에서 식별된 값에서 출발
N0_INIT = 0.05

# --- 학습
EPOCHS = 400
PATIENCE = 40
LR = 1e-3
HIDDEN = 64
BATCH = 2048
N_MC = 64
W_STATE = 2.0                 # 미래 상태(헤더·압력강하·입구온도) 지도
W_M = 1.0                     # 미래 유량 지도 (관측된 곳만)
W_PHYS = 1.0                  # 동시점 물리 정합 (유량 관측된 곳만)
RANDOM_SEED = 42

# --- 정제 단계 (2026-09-13). AE·KF·EKF 는 **정제용**이다 — 출력은 신경망 입력으로만 들어간다.
#   물리 잔차는 여전히 열수지 유량 관측 시각에만 건다(EKF 유량 추정으로 대체하지 않는다).
REFINE: tuple = ("kf", "ae", "ekf")   # 기본 = 정제 3종 모두 (문서 17). () 이면 문서 16 재현
REFINED_KFAE = {t: PROJECT_ROOT / "risk" / "output" / f"refined_{t}{RES_TAG}.parquet" for t in TRAINS}
REFINED_EKF = {t: OUTPUT_DIR / f"ekf_{t}{RES_TAG}.parquet" for t in TRAINS}
REFINE_COLS = {
    "kf": ["kf_t61", "innov_t61", "kf_hdr", "innov_hdr", "kf_tin", "innov_tin", "innov_t61_absmax"],
    "ae": ["ae_err", "ae_err_max"],
    "ekf": ["ekf_logm", "ekf_logm_sd", "innov_ekf_t61", "nis_ekf_t61"],
}


# --- Optuna · 8:2 시간분할 CV (2026-09-13) — lstm/test 와 같은 틀
#   시간순 8:2 (개발 80% / 시험 20%, 셔플 금지, 퍼지) → 개발구간만 TimeSeriesSplit 5-fold 로 탐색
#   → 최적값으로 개발구간 앞 90% 학습 · 뒤 10% 조기종료·등장성 보정 → 시험 20% 1회 평가.
#   ⚠ 시험구간이 문서 16~18(2022~)과 달라진다(≈2023-08~). 이전 수치와 직접 비교하지 말 것.
OPTUNA_DIR = OUTPUT_DIR / f"optuna{RES_TAG}"
FOLD_DIR = OPTUNA_DIR / "folds"


# --- 구조·타깃 (2026-09-13, 문서 19)
#   hard = 출력이 물리식을 반드시 거친다(기존 RestrictedPINN) · soft = 물리는 손실 제약(표준 PINN)
#   nn   = soft 와 같은 신경망에서 물리 잔차만 뺀 대조군
#   min_raw = 기존 타깃(분 단위 최저, 센서 하한 이상값 포함) · min = 이상값 제거 · mean = 1시간 평균 최저
ARCHS = ("hard", "soft", "nn", "lstm", "softw")   # lstm = 시계열 LSTM(물리 없음) · softw = soft + 물리 가중치 탐색 범위 확장 (문서 21)
TARGETS = ("min_raw", "min", "mean")
T61_SENSOR_FLOOR = -29.5      # TI61M·TI61Z 분 단위 최저가 정확히 −29.8~−30.0 (측정 하한) 인 1분 급락. 사이 값 없음


def variant_dir(variant: str = "auto", smoke: bool = False, arch: str = "hard", target: str = "min_raw"):
    """산출 폴더. auto = 정제 조합도 Optuna 가 고른 실행, 그 외 v_<조합>.
    기존 구조·타깃(hard·min_raw)은 OPTUNA_DIR 바로 아래, 나머지는 <구조>_<타깃>/ 아래."""
    d = OPTUNA_DIR / "smoke" if smoke else OPTUNA_DIR
    if (arch, target) != ("hard", "min_raw"):
        d = d / f"{arch}_{target}"
    return d if variant == "auto" else d / f"v_{variant}"           # fold 경계별 AE·EKF 재적합 산출
TEST_RATIO = 0.2
N_SPLITS = 5
N_TRIALS = 25
INNER_VAL_RATIO = 0.1
SEARCH_EPOCHS, SEARCH_PATIENCE, SEARCH_N_MC = 120, 15, 32
FINAL_EPOCHS, FINAL_PATIENCE = 400, 40
