"""AE용 데이터 준비 — 누수차단 로직은 `lstm.test.data` 를 **재사용**한다.

같은 규칙을 두 번 구현하면 한쪽만 고쳐져서 조용히 어긋난다.
표본 로드·윈도우 생성·purge 분할·robust 스케일러는 전부 lstm 쪽 구현을 그대로 쓴다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from lstm.test import data as lstm_data

from . import config

logger = logging.getLogger(__name__)

load_sample = lstm_data.load_sample


# 표준화 스케일러는 lstm.test.data 에 공용으로 둔다 (KF 도 같은 것을 쓴다).
# robust 스케일러를 쓰면 안 되는 이유는 그쪽 docstring 참고.
Scaler = lstm_data.StandardScaler


def make_windows(df: pd.DataFrame, window: int | None = None
                 ) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """재구성용 윈도우 (N, L, F). AE 는 타깃이 입력 자신이라 y 를 따로 두지 않는다."""
    W = config.WINDOW if window is None else window
    X, _, t_start, _ = lstm_data.make_windows(df, window=W, horizon=config.HORIZON)
    logger.info("AE 윈도우 %s", X.shape)
    return X, t_start


def split_purged(n: int, purge: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """시간순 8:2 + purge. 윈도우가 L-1 만큼 겹치므로 경계에서 L 개를 버린다."""
    return lstm_data.split_purged(n, test_ratio=config.TEST_RATIO,
                                  purge=config.PURGE if purge is None else purge)
