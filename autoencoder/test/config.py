"""Autoencoder 파일럿 설정.

역할: **PINN 이전 정제 단계.** 예측기가 아니다.
  - 다변수 센서 윈도우를 저차원 잠재벡터로 압축했다가 복원한다.
  - 산출물 ①: 재구성된(=잡음 제거된) 신호 → PINN 입력
  - 산출물 ②: 잠재벡터 → PINN 의 압축 상태표현 후보
  - 산출물 ③: 재구성오차 → 불량/이상 구간 플래그 (PINN 학습에서 뺄 구간 식별)

표본·피처·분할·누수차단 규칙은 `lstm.test` 와 **동일**하게 맞춘다(비교 가능하도록).
다만 과제가 다르므로 두 가지가 바뀐다:
  - 타깃: 미래값(t+H)이 아니라 **입력 자신**(재구성). 따라서 HORIZON 개념이 없다.
  - 베이스라인: persistence 가 아니라 **PCA**. 선형 AE 와 등가라, PCA 를 못 이기면
    비선형 AE 를 쓸 이유가 없다.
"""
from __future__ import annotations

from pathlib import Path

from lstm.test import config as lstm_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "autoencoder" / "test" / "output"

# --- lstm.test 와 공유하는 설정 (중복 정의하지 않는다) ---------------------
TREND_PARQUET = lstm_config.TREND_PARQUET
ALARM_PARQUET = lstm_config.ALARM_PARQUET
SAMPLE_START = lstm_config.SAMPLE_START      # 2025-01-01
SAMPLE_END = lstm_config.SAMPLE_END          # 2025-03-01
FEATURES = lstm_config.FEATURES              # 7개 (TI21Y·htx_*·PI-D2P_norm 제외)
WINDOW = lstm_config.WINDOW                  # 180분
TEST_RATIO = lstm_config.TEST_RATIO          # 8:2
N_SPLITS = lstm_config.N_SPLITS              # 5-fold TimeSeriesSplit
SEED = lstm_config.SEED

# 재구성 과제라 예측지평이 없다. 윈도우가 L-1 만큼 겹치므로 purge 는 WINDOW 면 충분하다.
HORIZON = 0
PURGE = WINDOW

# --- Optuna ----------------------------------------------------------------
N_TRIALS = 25
TUNE_EPOCHS = 15
FINAL_EPOCHS = 60
EARLY_STOP_PATIENCE = 8

SEARCH_SPACE = {
    "latent_dim": [2, 4, 8, 16],       # 압축 정도 — PINN 에 넘길 상태 차원
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": (0.0, 0.3),
    "lr": (1e-4, 5e-3),
    "batch_size": [64, 128, 256],
    "weight_decay": (1e-6, 1e-3),
}

# PCA 베이스라인에서 비교할 성분 수 (latent_dim 후보와 맞춘다)
PCA_COMPONENTS = [2, 4, 8, 16]

SHAP_BACKGROUND = 100
SHAP_EXPLAIN = 200
