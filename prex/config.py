"""HTR-31P(가스히터 P호기) 고장예측 전처리 설정.

대상을 HTR-31P 한 대로 한정한다 (2026-09-09 결정).
태그 목록 출처: data/docs/_tables/태그마스터.csv (2026-09-07 스냅샷)
  — 설비='가스히터' 이면서 라인='H-31P' (+ MOV31P*/HS31P*/H31P* 접두 공통열) 전수 조회.
문서 참고: data/docs/03_시계열데이터_구조와품질.md (§2~5, §8 전처리 체크리스트),
           data/docs/04_고장이벤트_라벨분석.md (§2 알람 유형 분류),
           data/docs/06_물리식_출처정리.md (§2-3 HTR-31P는 진공식 — 수위계 LI-D1P 없음).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TREND_DIR = DATA_DIR / "Trend 데이터" / "AI_CV"
ALARM_DIR = DATA_DIR / "Alram 데이터"
ALARM_DI_DIR = ALARM_DIR / "DI(파일이 커서 태그 50개 단위로 분리함)"

OUTPUT_DIR = PROJECT_ROOT / "prex" / "output"

# ---------------------------------------------------------------------------
# Trend(1분 연속 시계열) — HTR-31P 및 공통 경계조건 AI 태그
# ---------------------------------------------------------------------------
TREND_TAGS = [
    "PI21X",   # 주배관 입구압 (경계조건, 전 설비 공통)
    "TI21Y",   # 히터 입구온도 1 (경계조건, 공통)
    "TI21Z",   # 히터 입구온도 2 (경계조건, 공통)
    "TI33P",   # H-31P 출구온도 (관측/타깃)
    "TI-D2P",  # H-31P 수조(Bath) 수온 (상태변수)
    "PI-D2P",  # H-31P 진공압력 (상태변수 — 진공식 히터 고유, 수위계 LI-D1P는 없음)
]

# ---------------------------------------------------------------------------
# 가스 질량유량(m_gas) 역산 입력 — Trend에 유량계(FI61x)가 전량 NULL이라 직접 못 얻는다.
# 대신 정압기(PCV-41P) 밸브 유동식(문서 06 §3-1 G1)으로 역산한다.
#   Q = Cv · f(z) · sqrt((P_in^2 - P_out^2) / (G·T))
# 배관이 히터→정압기 직렬 연결(중간 저장/분기 없음)이므로 정압기 통과유량 ≈ 히터 통과유량(질량보존).
# 근거: 문서 05 §3-1(계통 구조), §4-1(정압기 계통), 06 §3(정압기 물리식).
FLOW_INPUT_TAGS = [
    "ZI41P",   # PCV-41P(Worker) SLEEVE VALVE 개도 지시 — f(z)의 z
    "PI43O",   # 정압기 출구 헤더압 — 발전계열(O/P/Q 공용) — P_out. PI43P는 KPOS에 없음(계열 공용 헤더로 계측)
    "RSF41P",  # PIC-41P 설정압(SP) — 유량식엔 안 쓰지만 e=PI43x-RSF41x 등 QA/검증용으로 같이 보관
]

# ⚠ 아래 상수는 전부 1차 근사치다. 사용 전 반드시 검증할 것 (문서 04 §5 미해결 항목 1·5):
#   - GAS_SG, PCV41P_CV: 영종 P호기 정압기 자체 도서가 아니라 남사(타 사이트) 승인도서의 동일 모델
#     (Fiorentini REFLUX 819) 카탈로그 값 — T3(proxy) 등급. 영종 자체 값 확인 필요.
#   - f(z): 실제 밸브 특성곡선(카탈로그 Fig.5) 대신 선형(z_pct) 근사. 과소/과대평가 가능.
#   - ZI41P_RESCALE_ABOVE: ZI41P는 2023~2024년 구간만 스팬이 0~100 -> 0~500으로 바뀐다(실측 확인,
#     연도별 median: ~2022 100 -> 2023 혼재 -> 2024 ~499 -> 2025 100). 정확한 교체일 미확인이므로
#     100 초과값은 전부 0~500 스팬으로 보고 5로 나눠 0~100%로 되돌리는 휴리스틱을 쓴다.
GAS_SG = 0.6                 # 천연가스 비중(공기=1) — GROVE/PF 카탈로그 공통값 (문서 06 §2-2, §3-3)
PCV41P_CV = 250.0            # Fiorentini REFLUX 819 Cv (문서 06 §3-3, 남사 승인도서 T3 proxy)
ZI41P_RESCALE_ABOVE = 100.0  # 이 값을 넘으면 0~500 스팬으로 보고 /5

# PI-D2P는 계기 스팬이 시기별로 바뀐다 (문서 03 §4):
#   2015~2024 구간만 스팬이 좁고(p99 -13~-85), 2011~2014·2025~ 구간은 넓다(-380~-558).
# 정확한 계기 교체일은 미확인 상태이므로 연 단위 근사 경계를 쓰고, era-wise z-score로 정규화한다.
PI_D2P_ERAS = [
    ("2011-2014", "2011-01-01", "2015-01-01"),
    ("2015-2024", "2015-01-01", "2025-01-01"),
    ("2025-", "2025-01-01", None),
]

# 결측이 이 시간 이상 이어지면 별개의 학습 시퀀스(segment)로 취급한다 (문서 03 §2-3, §8-5).
SEGMENT_GAP_HOURS = 24.0

# ---------------------------------------------------------------------------
# Alarm(DI/DO 이벤트 로그) — HTR-31P 관련 태그
# ---------------------------------------------------------------------------

# 핵심 고장성 알람 (문서 04 §2 분류 기준 — 예측 가치 ★★ 이상)
DI_FAULT_TAGS = {
    "BAB1P":   "FLAME DETECTION ALARM (화염검출 실패)",
    "TALLD1P": "BATH WATER TEMP. LOW LOW (수조 수온 극저)",
    "TALD1P":  "BATH WATER TEMP. LOW",
    "TAHD1P":  "BATH WATER TEMP. HIGH",
    "PAHHA1P": "FUEL GAS PRESS. HIGH HIGH",
    "PAHA1P":  "FUEL GAS PRESS. HIGH",
    "GAB1P":   "BURNER TROUBLE (FG/AIR)",
    "GLAB1P":  "GAS LEAKAGE ALARM",
    "GAD1P":   "BATH WATER LEVEL/TEMP 이상",
    "ZAF31P":  "MOV-31P FAULT ALARM (입출구 전동밸브 고장)",
}

# 밸브/운전 상태 (알람은 아니지만 문맥 판단에 필요 — 예: H31POH로 가동중 여부 확인)
DI_STATUS_TAGS = {
    "H31POH":   "H-31P OPERATION HEATER (가동상태)",
    "MOV31PO":  "MOV-31P OPENED",
    "MOV31PC":  "MOV-31P CLOSED",
    "MOV31PR":  "MOV-31P RUNNING",
    "MOV31PE":  "MOV-31P EXEC RUNNING",
    "MOV31POR": "MOV-31P OPEN RUNNING STATUS",
    "MOV31PCR": "MOV-31P CLOSE RUNNING STATUS",
    "CONF31P":  "MOV-31P CONFIRMED",
    "V31P":     "밸브상태 (IO LIST 미등재)",
    "V32P":     "밸브상태 (IO LIST 미등재)",
    "V33P":     "밸브상태 (IO LIST 미등재)",
}

DI_TAGS = {**DI_FAULT_TAGS, **DI_STATUS_TAGS}

# 운전원 원격/로컬 조작 명령 (DO) — 고장은 아니지만 개입 이력으로 유용
DO_TAGS = {
    "HS31PLS":  "MOV-31P SELECT(LOCAL)",
    "HS31PRS":  "MOV-31P SELECT(REMOTE)",
    "H31PERLS": "EMERGENCY RESET(LOCAL)",
    "H31PESLS": "EMERGENCY STOP(LOCAL)",
    "H31PARLS": "ALARM RESET(LOCAL)",
    "H31PERRS": "EMERGENCY RESET(REMOTE)",
    "H31PESRS": "EMERGENCY STOP(REMOTE)",
    "H31PARRS": "ALARM RESET(REMOTE)",
}

ALARM_TAG_META = {
    tag: {"description": desc, "category": cat}
    for cat, group in (("FAULT", DI_FAULT_TAGS), ("STATUS", DI_STATUS_TAGS), ("CONTROL", DO_TAGS))
    for tag, desc in group.items()
}

# 데이터 실측 범위 밖(2057/2058/2064년 등 KPOS 오류값)을 걸러내는 경계 (문서 03 §6-3)
ALARM_TIME_MIN = pd.Timestamp("2011-01-01")
ALARM_TIME_MAX = pd.Timestamp("2026-12-31")
