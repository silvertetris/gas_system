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

# ---------------------------------------------------------------------------
# 분석용 확장 컬럼 (2026-09-11 신설)
# ---------------------------------------------------------------------------
# TREND_TAGS(원본 6개)만 보면 정작 **모델이 쓰는 값**이 EDA에 안 나온다.
# 밸브식 입력(ZI41P/PI43O/RSF41P)과 prex 파생컬럼(ε·NTU 등)까지 합친 16개를
# 상관·결측 분석에만 쓴다. 개별 패널 그림(01~03)은 6개로 유지 — 16패널은 못 읽는다.
# 집합·순서는 prex/profiling.py 의 상관행렬과 동일하게 맞춰 뒀다(교차 확인 편의).
FLOW_TAGS = list(prex_config.FLOW_INPUT_TAGS)          # ZI41P, PI43O, RSF41P
DERIVED_TAGS = [
    "PI-D2P_norm",      # era-wise z-score (원본 PI-D2P 대신 쓰라고 만든 것)
    "ZI41P_frac",       # 개도 0~1 (스팬 보정 후)
    "gas_flow_proxy",   # ⚠ 약한 부하지표로 강등 (docs/htr31p_flow.md §7-2)
    "control_error_p",  # PI43O − RSF41P (정압기 제어오차)
    "htx_drive_c",      # T_bath − T_in : 구동 온도차 = ε의 분모
    "htx_eps",          # 열교환 효율 ε — 모델의 관측 z₃
    "htx_ntu",          # NTU = −ln(1−ε) = U·A/(m_gas·c_p) — 물리식 H3 좌변
]
ANALYSIS_TAGS = TREND_TAGS + FLOW_TAGS + DERIVED_TAGS   # 16개

TAG_LABELS: dict[str, str] = {
    "PI21X": "주배관 입구압\n(PI21X)",
    # TI21Y/Z는 이중화가 아니라 서로 다른 계열의 필터 출구 헤더다 (P&ID 확정, prex.config 주석).
    "TI21Y": "A/B계열 헤더온도\n(TI21Y, P호기 무관)",
    "TI21Z": "H-31P 입구온도\n(TI21Z)",
    "TI33P": "H-31P 출구온도\n(TI33P)",
    "TI-D2P": "수조 수온\n(TI-D2P)",
    "PI-D2P": "진공압력\n(PI-D2P)",
    # 확장 컬럼 (ANALYSIS_TAGS 전용 — 01~03 개별 패널에는 안 쓴다)
    "ZI41P": "정압기 개도\n(ZI41P)",
    "PI43O": "정압기 출구압\n(PI43O, 41O라인)",
    "RSF41P": "정압기 설정압\n(RSF41P)",
    "PI-D2P_norm": "진공압 정규화\n(PI-D2P_norm)",
    "ZI41P_frac": "개도 0~1\n(ZI41P_frac)",
    "gas_flow_proxy": "유량 프록시\n(gas_flow_proxy)",
    "control_error_p": "제어오차\n(PI43O−RSF41P)",
    "htx_drive_c": "구동 온도차\n(T_bath−T_in)",
    "htx_eps": "열교환 효율 ε",
    "htx_ntu": "NTU\n(=U·A/(m·c_p))",
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

# 상관 히트맵의 Spearman 패널용 표본수 (순위계산이 무거워 전체를 쓰지 않는다).
# prex/profiling.py 의 SPEARMAN_SAMPLE 과 같은 취지.
CORR_SPEARMAN_SAMPLE = 200_000
