"""prex 산출물(Parquet) 로드."""
from __future__ import annotations

import logging

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def load_trend() -> pd.DataFrame:
    """1분 그리드 연속 시계열(trend_htr31p.parquet)을 읽는다. Time을 index로 세팅."""
    if not config.TREND_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.TREND_PARQUET} 가 없습니다. 먼저 `python -m prex.pipeline` 을 실행하세요."
        )
    df = pd.read_parquet(config.TREND_PARQUET)
    df = df.set_index("Time").sort_index()
    logger.info("Trend 로드: %d행 (%s ~ %s)", len(df), df.index.min(), df.index.max())
    return df


def load_alarm_events() -> pd.DataFrame:
    """정제된 전체 DI/DO 이벤트(long, SET/RESET 모두 포함)를 읽는다."""
    if not config.ALARM_EVENTS_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.ALARM_EVENTS_PARQUET} 가 없습니다. 먼저 `python -m prex.pipeline` 을 실행하세요."
        )
    df = pd.read_parquet(config.ALARM_EVENTS_PARQUET)
    logger.info("Alarm 이벤트 로드: %d행", len(df))
    return df


def load_fault_events() -> pd.DataFrame:
    """상승엣지(발생 시점)만 추출된 이벤트표를 읽는다."""
    if not config.FAULT_EVENTS_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.FAULT_EVENTS_PARQUET} 가 없습니다. 먼저 `python -m prex.pipeline` 을 실행하세요."
        )
    df = pd.read_parquet(config.FAULT_EVENTS_PARQUET)
    logger.info("Fault(상승엣지) 이벤트 로드: %d행", len(df))
    return df
