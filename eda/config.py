"""태그별 EDA 설정."""
from __future__ import annotations

from pathlib import Path

from prex import config as p_config

PROJECT_ROOT = p_config.PROJECT_ROOT
RAW_TREND_DIR = p_config.TREND_DIR
OUTPUT_DIR = PROJECT_ROOT / "eda" / "output"
FIGURE_DIR = OUTPUT_DIR / "tags"

FIG_DPI = 110
FONT_CANDIDATES = ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]

# 정기점검 7건 (docs/05) — 시계열에 표시한다
MAINTENANCE = ["2013-03-28", "2015-03-16", "2017-03-31", "2019-03-21",
               "2021-04-02", "2022-10-21", "2024-10-29"]

RESAMPLE = "1D"          # 15년 전체를 그리려면 일 단위로 줄여야 읽힌다
HIST_BINS = 60
SAMPLE_MAX = 400_000     # 히스토그램·분포용 표본 상한
DEAD_NONZERO_MAX = 20_000   # 0 아닌 값이 이보다 적으면 DEAD 로 표시
