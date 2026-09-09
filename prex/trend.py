"""HTR-31P Trend(1분 연속 시계열) 전처리.

원본: data/Trend 데이터/AI_CV/*.csv (월별 175개 파일)
체크리스트 출처: data/docs/03_시계열데이터_구조와품질.md §8
  1. Time + 대상 태그만 읽기
  2. Time → datetime, NaT 제거
  3. Time 기준 중복 제거 (keep='last')
  4. 1분 그리드로 reindex → 결측은 명시적 NaN (보간 금지)
  5. 24시간 이상 결측은 학습 시퀀스 경계(segment)로 표시
  6. (해당 없음 — TI33C 등 죽은 태그는 애초에 TREND_TAGS에 없음)
  7. PI-D2P는 era-wise 재정규화
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _read_one_month(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["Time", *config.TREND_TAGS],
        dtype={tag: "float64" for tag in config.TREND_TAGS},
    )
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df.dropna(subset=["Time"])


def load_trend_raw() -> pd.DataFrame:
    """175개 월별 CSV를 읽어 하나로 합친다 (Time 파싱·NaT 제거까지)."""
    paths = sorted(config.TREND_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Trend CSV를 찾을 수 없습니다: {config.TREND_DIR}")
    frames = [_read_one_month(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    logger.info("Trend raw 로드: 파일 %d개, %d행", len(paths), len(df))
    return df


def dedup_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """같은 Time 중복 시 마지막 값 유지 (2019년 이후 주 단위 재기록 중복 대응)."""
    before = len(df)
    df = df.sort_values("Time").drop_duplicates(subset="Time", keep="last")
    logger.info("Time 중복 제거: %d -> %d행", before, len(df))
    return df


def compute_segments(observed_time: pd.Series, gap_hours: float = config.SEGMENT_GAP_HOURS) -> pd.Series:
    """실측 시각들 사이 gap_hours 이상 결측이면 새 segment로 취급.

    Returns: 실측 시각을 index로 하는 segment_id Series (0부터 시작).
    """
    t = observed_time.sort_values()
    boundary = t.diff() >= pd.Timedelta(hours=gap_hours)
    segment_id = boundary.cumsum()
    return pd.Series(segment_id.values, index=t.values, name="segment_id")


def build_minute_grid(df: pd.DataFrame) -> pd.DataFrame:
    """1분 간격 전체 그리드로 reindex. 결측은 보간하지 않고 명시적 NaN으로 남긴다."""
    seg = compute_segments(df["Time"])
    df = df.set_index("Time")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    grid = df.reindex(full_index)
    grid.index.name = "Time"

    # 그리드의 각 시각에, 그 시각 이전(또는 같은 시각)의 가장 최근 실측 경계를 전파한다.
    grid["segment_id"] = seg.reindex(grid.index, method="ffill")

    logger.info(
        "1분 그리드 생성: %d행 (%s ~ %s), segment %d개",
        len(grid), full_index.min(), full_index.max(), grid["segment_id"].nunique(),
    )
    return grid


def add_era_normalization(df: pd.DataFrame, column: str = "PI-D2P") -> pd.DataFrame:
    """계기 스팬이 시기별로 바뀐 태그에 대해 구간별(era) z-score 정규화 컬럼을 추가한다."""
    era = pd.Series(pd.NA, index=df.index, dtype="object")
    for label, start, end in config.PI_D2P_ERAS:
        mask = df.index >= pd.Timestamp(start)
        if end is not None:
            mask &= df.index < pd.Timestamp(end)
        era.loc[mask] = label
    df[f"{column}_era"] = era

    normed = pd.Series(np.nan, index=df.index, dtype="float64")
    for label in era.dropna().unique():
        mask = era == label
        vals = df.loc[mask, column]
        mu, sigma = vals.mean(), vals.std()
        if sigma and not np.isnan(sigma):
            normed.loc[mask] = (vals - mu) / sigma
    df[f"{column}_norm"] = normed
    return df


def run_trend_pipeline() -> pd.DataFrame:
    """Trend 전처리 전체 실행: 로드 → 중복제거 → 1분 그리드 → era 정규화."""
    df = load_trend_raw()
    df = dedup_by_time(df)
    grid = build_minute_grid(df)
    grid = add_era_normalization(grid, column="PI-D2P")
    return grid
