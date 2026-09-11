"""LSTM 파일럿 설정.

목적: KF-PINN 비교군으로 쓸 LSTM 파이프라인을 **표본 구간에서 먼저 검증**한다.
      전체 데이터(7.8M행)가 아니라 아래 SAMPLE_START~END 구간만 쓴다.

설계 근거는 docs/kfpinn_state_space.md(태그↔변수) 와 docs/htr31p_flow.md §7~§8:
  - T_in 은 TI21Z 단독 (TI21Y 는 인천도시가스 계열이라 P호기와 무관 — §7-1)
  - htx_eps/htx_ntu 는 입력 태그 3개의 함수라 정보 중복 → 제외
  - PI-D2P_norm 은 era 전체(=테스트 구간 포함) 통계로 정규화돼 **누수** → 원본 PI-D2P 사용
  - gas_flow_proxy 는 ZI41P_frac 과 Pearson 0.984 로 사실상 중복 → ZI41P_frac 만 사용
"""
from __future__ import annotations

from pathlib import Path

from prex import config as prex_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TREND_PARQUET = prex_config.OUTPUT_DIR / "trend_htr31p.parquet"
ALARM_PARQUET = prex_config.OUTPUT_DIR / "alarm_events_htr31p.parquet"
OUTPUT_DIR = PROJECT_ROOT / "lstm" / "test" / "output"

# ---------------------------------------------------------------------------
# 표본 구간 — 전체의 약 1.1%
# ---------------------------------------------------------------------------
# 선정 기준(2026-09-11 실측): 결측 0%, 단일 segment, 가동률 99.6%, 86,400행.
# deep-2(2024-11~) 체제에서 뽑았다 — 현재 운전 방식이고 가동률이 가장 높아 학습 신호가 많다.
# ⚠ 다른 체제(shallow)로 일반화된다는 보장은 없다. 파일럿이므로 한 체제 안에서만 검증한다.
SAMPLE_START = "2025-01-01"
SAMPLE_END = "2025-03-01"

# ---------------------------------------------------------------------------
# 피처 / 타깃
# ---------------------------------------------------------------------------
FEATURES = [
    "TI21Z",       # 히터 입구온도 (T_in) — 외생 입력
    "TI33P",       # 히터 출구온도 (T_out) — 타깃의 과거값(자기회귀 입력)
    "TI-D2P",      # 수조 수온 (T_bath) — 상태
    "PI-D2P",      # 수조 진공압 (원본. _norm 은 누수라 안 씀)
    "PI21X",       # 주배관 입구압 — 경계조건
    "ZI41P_frac",  # 정압기 개도(0~1) — 부하 대리변수
    "burner_on",   # H31POH 를 1분 그리드에 전파한 0/1 (data.py 에서 생성)
]
TARGET = "TI33P"

# ---------------------------------------------------------------------------
# 윈도우 / 예측지평 — 물리 시정수 기준
# ---------------------------------------------------------------------------
# 수조 시정수 τ = M_w·c_w / U·A ≈ 63분 (prex.config.HTR_BATH_TIME_CONST_S).
#   입력 윈도우 L = 180분 ≈ 3τ,  예측지평 H = 60분 ≈ 1τ.
# ⚠ H=1분은 쓰지 말 것 — persistence(직전값 복사)가 이겨서 학습이 무의미해진다.
WINDOW = 180
HORIZON = 60

# ---------------------------------------------------------------------------
# 분할 / 교차검증
# ---------------------------------------------------------------------------
TEST_RATIO = 0.2          # 시간순 8:2 (셔플 금지)
N_SPLITS = 5              # Optuna 안에서 쓰는 TimeSeriesSplit fold 수
# 누수 차단용 purge 간격. 윈도우 길이+지평만큼 버리면 학습 윈도우의 타깃 시각과
# 검증 윈도우의 입력 구간이 절대 겹치지 않는다.
PURGE = WINDOW + HORIZON

# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
N_TRIALS = 25
TUNE_EPOCHS = 15          # 탐색 중에는 짧게
FINAL_EPOCHS = 60         # 최종 학습은 길게 + early stopping
EARLY_STOP_PATIENCE = 8

SEARCH_SPACE = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": (0.0, 0.4),
    "lr": (1e-4, 5e-3),          # log scale
    "batch_size": [64, 128, 256],
    "weight_decay": (1e-6, 1e-3),  # log scale
}

SEED = 42
SHAP_BACKGROUND = 100     # SHAP 배경 표본 수
SHAP_EXPLAIN = 200        # 설명할 테스트 표본 수
