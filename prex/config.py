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
    "TI21Y",   # ⚠ P호기와 무관 — 인천도시가스(A/B) 계열 헤더. 대조용으로만 보관 (아래 참조)
    "TI21Z",   # H-31P 입구온도 (T_in) — 인천공항·발전(O/P) 계열 헤더. P&ID 확정
    "TI33P",   # H-31P 출구온도 (관측/타깃)
    "TI-D2P",  # H-31P 수조(Bath) 수온 (상태변수)
    "PI-D2P",  # H-31P 진공압력 (상태변수 — 진공식 히터 고유, 수위계 LI-D1P는 없음)
]

# ---------------------------------------------------------------------------
# 히터 입구온도 T_in (2026-09-11 P&ID로 확정 — docs/htr31p_flow.md §7-1)
# ---------------------------------------------------------------------------
# IO LIST는 TI21Y·TI21Z를 둘 다 "HEATER INLET TEMP."로만 적어 놔서 라인 배정을 알 수 없었고,
# 그동안은 둘을 평균해서 T_in으로 썼다. P&ID(도면 09-15-R-32-001/004)를 판독한 결과
# **둘은 이중화가 아니라 서로 다른 수요처 계열의 필터 출구 헤더**임이 확인됐다:
#
#   TI21Z  ← 배관 400-NG-P7-210 : F-21O + F-21P 출구 합류 → HTR-31O · HTR-31P   ✅ P호기
#   TI21Y  ← 배관 350-NG-P7-211 : F-21A + F-21B 출구 합류 → 도면4 350-NG-P7-301
#                                  → 250-NG-P7-302-H/303-H → HTR-31A · HTR-31B  ❌ 도시가스 계열
#
# 따라서 평균은 인천도시가스 계열 온도를 P호기에 섞는 오염이었다. T_in = TI21Z 단독으로 쓴다.
# (실측 corr 0.87 / p99 4.5℃ 차이도 "이중화가 아니다"로 설명된다 — 문서 03 §3-1.)
# TI21Y는 파이프라인에서 계산에 쓰지 않되, 계열 간 대조·QA용으로 parquet에는 계속 남긴다.
HTR31P_INLET_TAG = "TI21Z"
HTR31P_INLET_TAG_UNUSED = "TI21Y"  # A/B 계열 — 계산에 쓰지 말 것

# ---------------------------------------------------------------------------
# 가스 질량유량(m_gas) 역산 입력 — Trend에 유량계(FI61x)가 전량 NULL이라 직접 못 얻는다.
# 대신 정압기(PCV-41P) 밸브 유동식(문서 06 §3-1 G1)으로 역산한다.
#   Q = Cv · f(z) · sqrt((P_in^2 - P_out^2) / (G·T))
#
# ⚠⚠ 2026-09-11 P&ID 판독으로 이 역산의 전제가 무너졌다 (docs/htr31p_flow.md §7-2).
#   기존 전제: "히터→정압기 직렬 연결(중간 저장/분기 없음) → 정압기 통과유량 ≈ 히터 통과유량"
#   실제 배관: HTR-31O 출구(400-NG-P7-354-H) + HTR-31P 출구(400-NG-P7-355)가 **공통 헤더**
#             400-NG-P7-358 로 합류한 뒤 → 도면2 350-NG-P7-451 → 정압기 41O/41P/41Q/41R **4라인 분기**.
#   즉 히터 2대 : 정압기 4라인이며 1:1 대응이 아니다. PCV-41P 통과유량 ≠ HTR-31P 통과유량.
#   → gas_flow_proxy는 "P호기 히터 통과유량"이 아니라 **"O/P 공통헤더에서 41P 라인이 뽑아 간 유량"**이다.
#     H2식(m_gas·c_p·ΔT = U·A·ΔT_lm)의 m_gas 대용으로 쓰면 편향된다. 문서 04 §5-1의 대안대로
#     m_gas는 KF-PINN 잠재변수로 공동추정하는 쪽으로 옮기는 것을 권장한다.
#   가로지르는 크로스타이도 있다: 350-NG-P7-319 (O/P 헤더 ↔ 도면4 A/B측, '98.08.26 "ADD HEATER
#   COMMON LINE" 개정) — 계열 간 완전 독립도 아니다.
# 근거: P&ID 09-15-R-32-001(히터), -002(정압기 O/P/Q/R), -004(히터 A/B).
FLOW_INPUT_TAGS = [
    "ZI41P",   # PCV-41P(Worker) SLEEVE VALVE 개도 지시 — f(z)의 z
    # PI43O = P_out. ⚠ "O/P/Q 공용 헤더"가 아니다 — P&ID상 PT-43O는 PCV-41O/42O **유닛 전용 출구배관**
    # 360-NG-P7-456(V-42O 직전)에 달려 있고, 3라인 합류는 그보다 하류인 600-NG-P3-460/461이다.
    # 41P 유닛에는 로컬 게이지 PI-43P가 있으나 전송기(PT)가 없어 SCADA에 안 올라온다 → KPOS에 PI43P 없음.
    # 즉 PI43O는 "옆 라인(41O)의 출구압"이며, 대용 근사의 강도가 종전 가정보다 한 단계 세다.
    "PI43O",
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

# ---------------------------------------------------------------------------
# PI-D2P 진공압 regime (2026-09-11 전면 재검토 — docs/htr31p_flow.md §8)
# ---------------------------------------------------------------------------
# 기존 해석: "계기 스팬이 2015년과 2025년에 바뀌었다" → 연 단위 근사 경계 + era-wise z-score.
# 실측 재확인 결과 **경계가 2개월씩 틀렸고, 성격도 계기가 아니라 설비/운전 상태로 보인다**:
#
#   2011-10 ~ 2014-10-20  깊은 진공 (중앙 -530~-658)   37개월
#   2014-10-21 ~ 2024-10  얕은 진공 (중앙  -33~-216)  120개월  ← 하루 만에 계단 전이
#   2024-11-01 ~ 2026-08  깊은 진공 (중앙 -600대)      22개월  ← 13일 정지정비(10/17~29) 직후
#
# 전이가 설비 상태라는 근거: 2024-11 복구와 **동시에** 히터 승온폭(TI33P−TI21Z) 중앙값이
# 1.8℃ → 31℃, 버너 ON 비율이 18% → 34%로 함께 뛴다. 계기 스팬만 바뀐 거라면 TI33P는 안 변한다.
# ⚠ 단 2014-10-21 전이는 수조온도 변화 없이(17.60→17.48℃) 하루 만에 -649→-113으로 튀어
#    계기 교체 가능성도 남아 있다. 정기점검 이력 파싱 전까지 **둘 다 열어 둔다**.
#
# ⚠⚠ `PI-D2P_norm`(era-wise z-score) 사용 주의: 구간별로 평균 0으로 맞추므로 위 regime 차이를
#    **지워 버린다**. "계기 스팬 보정"이라는 원래 명분이 약해졌으므로, 열화·이상탐지 목적이면
#    원본 `PI-D2P` 또는 `vacuum_regime` 라벨을 쓰고 `_norm`은 쓰지 말 것.
PI_D2P_ERAS = [
    ("deep-1", "2011-01-01", "2014-10-21"),    # 깊은 진공 (정상)
    ("shallow", "2014-10-21", "2024-11-01"),   # 얕은 진공 (10년)
    ("deep-2", "2024-11-01", None),            # 깊은 진공 (복구)
]
# regime 라벨을 만드는 임계값. 월 중앙값 기준으로 -400 mmHg 를 경계로 두면 위 3구간이 그대로 나온다.
VACUUM_DEEP_THRESHOLD = -400.0

# 결측이 이 시간 이상 이어지면 별개의 학습 시퀀스(segment)로 취급한다 (문서 03 §2-3, §8-5).
SEGMENT_GAP_HOURS = 24.0

# ---------------------------------------------------------------------------
# HTR-31P 히터 설계계수 (제작도서 OCR 추출, 2026-09-10)
#   출처: `경인 22-영종관리소진공식 가스히터(HTR-31P)-제작도서.pdf` 열전달 계산서 SHEET 1~3
#         (원일티엔아이 WONIL, PO-09-029, 2009.07.06 승인). 신뢰도 T1(1차 원본).
#   환산·상세 근거: docs/htr31p_flow.md §5.
#   ⚠ 이 값들은 "설계점(design point)" 상수다. 운전 중 실제 U는 오염·열화로 변하고(그게 예측 대상)
#      수조 열용량 M_w·c_w는 아직 미확보 → H1/H2에서 U·A는 초기값/prior로 쓰고 열화항으로 보정할 것.
# ---------------------------------------------------------------------------
KCAL_TO_J = 4186.8           # 1 kcal = 4186.8 J  (kcal/h → W 는 ×KCAL_TO_J/3600)

# 열전달 (튜브번들: 공정가스 ↔ 열매체)
HTR_U_KCAL = 513.6           # 총괄전열계수 U [kcal/m²·h·℃] (내막 2048 / 외막 863.1 / 오염계수 포함)
HTR_AREA_M2 = 27.1           # 전열면적 A [m²] (106본 × π × Ø25.4mm × L3200mm)
HTR_UA_W_PER_K = HTR_U_KCAL * HTR_AREA_M2 * KCAL_TO_J / 3600.0   # U·A ≈ 16,188 W/K (=16.2 kW/K)

# 설계점 열·물질수지
HTR_DESIGN_FLOW_NM3_H = 65_000.0     # 설계 가스 체적유량 [Nm³/h] (정압기 61,900과 정합)
HTR_GAS_DENSITY_KG_NM3 = 0.80683     # 가스 밀도 [kg/Nm³] (비중 0.624, Air=1)
HTR_DESIGN_MGAS_KG_S = HTR_DESIGN_FLOW_NM3_H * HTR_GAS_DENSITY_KG_NM3 / 3600.0   # ≈ 14.57 kg/s (=52,444 kg/h)
HTR_CP_GAS_J_KGK = 2739.8            # 가스 정압비열 c_p [J/kg·K] (=0.654 kcal/kg·℃; 문서 06 §2 근사 2.5보다 정밀)
HTR_BATH_TEMP_C = 80.0               # 열매체(수조) 설계온도 [℃]
HTR_DESIGN_LMTD_C = 68.19            # 설계 대수평균온도차 [℃] (bath 80 / gas 0→22.4℃)

# 설계 열량 (번들 흡수열) — 제작도서 설계기준
HTR_DESIGN_DUTY_KCAL_H = 860_000.0                              # 설계 흡수열 [kcal/h] (=0.86 Gcal/h)
HTR_DESIGN_DUTY_W = HTR_DESIGN_DUTY_KCAL_H * KCAL_TO_J / 3600.0  # ≈ 1.0 MW
#   교차검증: U·A·LMTD = 16,188 × 68.19 ≈ 1.10 MW ≈ 흡수열(1.0MW)+손실 → 설계와 정합.

# ⚠⚠ Q_burner 정정 (H1 구동항) --------------------------------------------------
#   기존 문서(06_물리식_출처정리.md §2)의 P=52 Gcal/h 는 제작도서 설계열량(0.86 Gcal/h)과 60배 불일치.
#   물리적으로도 65,000 Nm³/h 를 Δt≈22~35℃ 데우는 데 52 Gcal/h 는 불가능(fire tube 가 860,000 kcal/h
#   =34.4 m² 기준으로 설계됨; 52 Gcal/h 면 ~2,000 m² 필요). "52"는 설계 질량유량 52,444 kg/h 와 수치가
#   일치 → 영종설비현황.xlsx 에서 유량을 열량으로 오기입한 것으로 의심. 상세: docs/htr31p_flow.md §5-1.
#   → 버너 설계 발열은 제작도서 "CALCULATION SHEET FOR BURNER"(p278)의 연료 소비량으로 직접 얻는다:
#        Qg(연료소비) = Qc/Hℓg = 860,000/9,510 = 90 Nm³/h  →  발열 = 90 × 9,510 ≈ 856,000 kcal/h ≈ 0.86 Gcal/h.
HTR_FUEL_LHV_KCAL_NM3 = 9_510.0                                # 연료가스 저위발열량 Hℓg [kcal/Nm³] (제작도서 p278)
HTR_FUEL_CONSUMPTION_NM3_H = 90.0                              # 설계 연료 소비량 Qg [Nm³/h] (제작도서 p278, =Qc/Hℓg)
HTR_QBURNER_DESIGN_W = HTR_FUEL_CONSUMPTION_NM3_H * HTR_FUEL_LHV_KCAL_NM3 * KCAL_TO_J / 3600.0  # ≈ 0.996 MW
HTR_QBURNER_DESIGN_GCAL_H = HTR_QBURNER_DESIGN_W * 3600.0 / KCAL_TO_J / 1e6      # ≈ 0.86 Gcal/h
HTR_COMBUSTION_EFF = 0.85                                       # 연소효율(2015 시험 Effg) — 손실 고려 시 실연료 약간↑
#   주: 제작도서 계산은 연료 LHV heat ≈ 흡수열(860,000 kcal/h)로 잡음(현열, 손실 미분리) → 설계 버너발열 ≈ 0.86 Gcal/h.
#   실시간 Q_burner(t)는 여전히 미계측(연료 유량계 태그 없음) → H31POH(on/off) × 이 상수로 근사하거나 잠재변수.
# ❌ 폐기: Q_burner = 52 Gcal/h (영종설비현황.xlsx, 약 60배 오류 의심 — 사용 금지)

# 수조 열용량 M_w·c_w (H1의 dT_bath/dt 계수) — 제작도서 OCR 확보 (2026-09-10)
#   출처: WEIGHT ANALYSIS OF GAS HEATER(제작도서 p31 "30. LIQUID = 14,512 KG")
#         + CALCULATION OF VOLUMN(p40, 내부 액체 체적 TOTAL 14.527 m³).
#   교차검증: 14,512 kg ÷ 14.527 m³ = 999 kg/m³ ≈ 물 밀도 → 진공수조(열매체를 water로 모델링).
HTR_BATH_LIQUID_MASS_KG = 14_512.0                             # 열매체 충전량 M_w [kg] (체적 14.527 m³와 정합)
HTR_CW_J_KGK = 1.0 * KCAL_TO_J                                 # 열매체 비열 c_w [J/kg·K] (=1.0 kcal/kg℃, 계산서 WATER)
HTR_MW_CW_J_PER_K = HTR_BATH_LIQUID_MASS_KG * HTR_CW_J_KGK      # 수조 열용량 ≈ 60.76 MJ/K
HTR_BATH_TIME_CONST_S = HTR_MW_CW_J_PER_K / HTR_UA_W_PER_K      # τ = M_w·c_w / U·A ≈ 3,754 s (≈63분), H1 스케일 sanity

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

# ---------------------------------------------------------------------------
# 열교환기 효율 ε / NTU 산출 조건 (2026-09-11 신설 — docs/kfpinn_state_space.md §2)
# ---------------------------------------------------------------------------
# ε = (T_out − T_in)/(T_bath − T_in) 는 분모가 작아지면 발산한다. 구동 온도차가 이 값보다
# 작은 시각은 계산하지 않는다(NaN). 15℃는 실측 분포에서 ε가 안정되는 하한이고, 이 조건으로
# 전체 행의 약 64%가 유효하다(2026-09-11 재실행 기준 5,021,531행).
EPS_MIN_DRIVE_C = 15.0
# ε는 원리상 0<ε<1. 계기 dropout·온도 역전으로 범위를 벗어난 값은 버린다(보간하지 않음).
EPS_VALID_MIN = 0.01
EPS_VALID_MAX = 0.99
