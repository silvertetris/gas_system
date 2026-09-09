"""EDA 시각화 설정.

입력은 prex 파이프라인의 산출물(prex/output/*.parquet)이다.
태그 설명/분류는 prex.config를 그대로 재사용해 중복 정의를 피한다.
"""
from __future__ import annotations

from pathlib import Path

from prex import config as prex_config

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREX_OUTPUT_DIR = prex_config.OUTPUT_DIR

TREND_PARQUET = PREX_OUTPUT_DIR / "trend_htr31p.parquet"
ALARM_EVENTS_PARQUET = PREX_OUTPUT_DIR / "alarm_events_htr31p.parquet"
FAULT_EVENTS_PARQUET = PREX_OUTPUT_DIR / "fault_events_htr31p.parquet"

OUTPUT_DIR = PROJECT_ROOT / "trend" / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Trend 수치 태그 (prex.config.TREND_TAGS와 동일 집합, 표시용 한글 라벨만 추가)
# ---------------------------------------------------------------------------
TREND_TAGS = list(prex_config.TREND_TAGS)

TAG_LABELS: dict[str, str] = {
    "PI21X": "주배관 입구압\n(PI21X)",
    "TI21Y": "히터 입구온도1\n(TI21Y)",
    "TI21Z": "히터 입구온도2\n(TI21Z)",
    "TI33P": "H-31P 출구온도\n(TI33P)",
    "TI-D2P": "수조 수온\n(TI-D2P)",
    "PI-D2P": "진공압력\n(PI-D2P)",
}

# ---------------------------------------------------------------------------
# 알람 메타 (prex.config 재사용)
# ---------------------------------------------------------------------------
ALARM_TAG_META = prex_config.ALARM_TAG_META
CATEGORY_ORDER = ["FAULT", "STATUS", "CONTROL"]

# ---------------------------------------------------------------------------
# 그림 스타일 / 샘플링
# ---------------------------------------------------------------------------
FIG_DPI = 150
FONT_FAMILY_CANDIDATES = ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]

# 분포·산점도(pairplot)용 그림은 전체(수백만행)를 그리면 느리고 가독성도 떨어져 샘플링한다.
# 상관계수·히트맵 등 집계 통계는 전체 데이터로 계산한다(샘플링하지 않음).
DIST_SAMPLE_MAX = 200_000
PAIRPLOT_SAMPLE_MAX = 5_000

RANDOM_SEED = 42

CATEGORY_PALETTE = {
    "FAULT": "#d94f4f",
    "STATUS": "#4f8cd9",
    "CONTROL": "#7f7f7f",
}
