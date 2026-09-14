"""Kalman Filter 파일럿 설정.

역할: **PINN 이전 정제 단계.** AE 와 같은 위치지만 접근이 다르다.
  - AE: 데이터로부터 배운 비선형 압축 → 재구성
  - KF: **명시적 상태공간 모델 + 베이즈 갱신** → 상태 추정
  산출물 ①: 필터링된 상태 궤적(잡음 제거된 신호) → PINN 입력
  산출물 ②: innovation(예측오차) / NIS → 이상·불량 구간 플래그
  산출물 ③: Kalman gain → 각 계기를 얼마나 신뢰하는지 (계기 품질 진단)

⚠ **여기서는 물리식(H1~H4)을 넣지 않는다.** 물리는 PINN 이 담당하고, KF 는
   "잡음 섞인 계측에서 부드러운 상태를 뽑는" 정제 역할만 한다. 물리를 KF 에도 넣으면
   두 단계가 같은 가정을 중복으로 깔게 되고, PINN 이 추정해야 할 U·A 를 KF 가 먼저
   고정해 버린다. (물리 결합은 KF-PINN 통합 단계에서 한다 — docs/kfpinn_state_space.md)

**구조적 차이**: LSTM/AE 는 윈도우(N, L, F) 단위지만 KF 는 **시계열을 그대로** 따라간다.
   그래서 윈도우 생성이 없고, purge 도 단순 시간 간격이면 된다.
"""
from __future__ import annotations

from pathlib import Path

from lstm.test import config as lstm_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "kalman" / "test" / "output"

# --- lstm.test 와 공유 (표본·분할 동일하게) --------------------------------
TREND_PARQUET = lstm_config.TREND_PARQUET
ALARM_PARQUET = lstm_config.ALARM_PARQUET
SAMPLE_START = lstm_config.SAMPLE_START
SAMPLE_END = lstm_config.SAMPLE_END
TEST_RATIO = lstm_config.TEST_RATIO
N_SPLITS = lstm_config.N_SPLITS
SEED = lstm_config.SEED

# --- KF 대상 신호 ----------------------------------------------------------
# burner_on 은 0/1 이산이라 연속 상태로 필터링하면 안 된다 → 제외하고 외생입력으로만 본다.
SIGNALS = ["TI21Z", "TI33P", "TI-D2P", "PI-D2P", "PI21X", "ZI41P_frac"]

# 시계열이라 윈도우가 없다. 학습/테스트 사이를 이 시간만큼 비운다(경계 전이 영향 차단).
PURGE_MIN = 180
DT = 1.0          # 샘플 간격 [분] — prex 가 1분 그리드로 만들어 둠

# --- Optuna ----------------------------------------------------------------
# 튜닝 대상은 KF 의 고전적 하이퍼파라미터인 **프로세스 잡음 Q 와 관측 잡음 R** 이다.
# 신호를 robust 스케일링한 뒤 돌리므로 전역 스칼라 3개로 충분하다.
N_TRIALS = 25
SEARCH_SPACE = {
    "model_type": ["local_level", "local_trend"],   # 상태에 기울기를 둘 것인가
    "q_level": (1e-8, 1e-1),                        # log
    "q_slope": (1e-10, 1e-3),                       # log (local_trend 일 때만)
    "r_scale": (1e-4, 1e1),                         # log
}

# --- 베이스라인 -------------------------------------------------------------
# AE 의 PCA 에 해당. KF 가 이 단순 평활기들을 못 이기면 상태공간 모델을 쓸 이유가 없다.
EWMA_ALPHAS = [0.05, 0.1, 0.3, 0.5]
MA_WINDOWS = [5, 15, 60]
