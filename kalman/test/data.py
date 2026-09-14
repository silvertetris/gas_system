"""KF용 데이터 준비 — 시계열을 그대로 쓴다(윈도우 없음).

표본 로드·스케일러는 `lstm.test.data` 재사용. 분할만 KF 에 맞게 시간 기준으로 다시 정의한다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from lstm.test import data as lstm_data

from . import config

logger = logging.getLogger(__name__)

# ⚠ robust 가 아니라 **표준화** 스케일러를 쓴다.
# robust 로 돌렸더니(2026-09-11) ZI41P_frac 의 IQR 이 0.0018 이라 스케일 후 σ 가 121 이 되고,
# NIS 가 130,039(정상 1.0), 1스텝 RMSE 지표까지 그 신호가 지배했다.
Scaler = lstm_data.StandardScaler
load_sample = lstm_data.load_sample


def to_series(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """(T, S) 관측 행렬. 결측은 NaN 으로 남긴다 — KF 가 갱신을 건너뛰며 처리한다."""
    Z = df[config.SIGNALS].to_numpy(dtype="float64")
    logger.info("KF 입력 시계열 %s, 결측 %.2f%%", Z.shape, 100 * np.isnan(Z).mean())
    return Z, df.index


def split_purged(n: int, test_ratio: float = config.TEST_RATIO,
                 purge_min: int = config.PURGE_MIN) -> tuple[np.ndarray, np.ndarray]:
    """시간순 8:2 + 경계 purge.

    KF 는 인과 필터라 '미래를 본다'는 의미의 누수는 없다. 다만 **Q/R 튜닝**이
    테스트 구간 통계를 보면 안 되므로 학습 구간을 명확히 자르고, 경계 전이의 영향을
    피하려고 purge_min 분을 버린다.
    """
    cut = int(n * (1 - test_ratio))
    train = np.arange(0, max(cut - purge_min, 0))
    test = np.arange(min(cut + purge_min, n), n)
    logger.info("분할: train %d / purge %d(버림) / test %d",
                len(train), n - len(train) - len(test), len(test))
    return train, test
