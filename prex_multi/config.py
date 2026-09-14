"""4대 히터 공통 설정.

왜 확장하나 (docs/htr31p_flow.md §11-4):
  HTR-31P 는 **진공식(감압비등)** 이라 수조 유효 열용량이 미지다(현열의 3.0배로 추정).
  이 미지수 때문에 `Q_loss` 가 음수로 나오는 등 추정이 닫히지 않았다.
  **A/B/O 는 대기압식이라 `C_eff = M_w·c_w` 로 고정**되므로, 같은 파이프라인을
  거기서 먼저 검증하고 P 로 되돌릴 수 있다.

부수적 이점:
  - A/B/O 에는 **수조 수위계**(`LI-D1A/B/O`)가 있다. P 는 없다(§문서 03 §3-1).
    열매체 누수·보충은 기계고장의 17%(§10-2)라 수위 관측은 가치가 크다.
  - 표본이 4배가 되고, 문서 04 §2-1 이 "가장 예측 가치 높은 구간"으로 꼽은
    2024~2025 H-31A/B 화염검출 급증(A 422·393건, B 379·354건)이 들어온다.

⚠ 입구온도 매핑은 **P&ID 로 확정된 것**을 쓴다(§7-1):
    A·B ← TI21Y  (F-21A/B 출구 합류 헤더, 350-NG-P7-211)
    O·P ← TI21Z  (F-21O/P 출구 합류 헤더, 400-NG-P7-210)
  이걸 틀리면 다른 계열 온도를 넣게 된다.
"""
from __future__ import annotations

from prex import config as p_config

# 경로·공통 상수는 prex.config 재사용 (중복 정의 금지)
TREND_DIR = p_config.TREND_DIR
ALARM_DIR = p_config.ALARM_DIR
ALARM_DI_DIR = p_config.ALARM_DI_DIR
OUTPUT_DIR = p_config.PROJECT_ROOT / "prex_multi" / "output"
KCAL_TO_J = p_config.KCAL_TO_J
SEGMENT_GAP_HOURS = p_config.SEGMENT_GAP_HOURS

UNITS = ["A", "B", "O", "P"]

# ---------------------------------------------------------------------------
# 히터별 태그 (전부 Trend/Alarm 원본에 존재함을 2026-09-12 확인)
# ---------------------------------------------------------------------------
INLET_TAG = {"A": "TI21Y", "B": "TI21Y", "O": "TI21Z", "P": "TI21Z"}   # P&ID §7-1

def trend_tags(u: str) -> dict[str, str]:
    """히터 u 의 Trend(AI) 태그. 없는 것은 None."""
    return {
        "T_in":   INLET_TAG[u],
        "T_out":  f"TI33{u}",
        "T_bath": f"TI-D2{u}",
        "level":  f"LI-D1{u}" if u != "P" else None,   # P 는 수위계 미설치
        "vacuum": "PI-D2P" if u == "P" else None,      # P 만 진공식
    }

# 연료가스압만 A/B 가 공용이다(PAHHA1M). 나머지 9종은 4대 전부 개별 태그가 있다.
def alarm_tags(u: str) -> dict[str, str]:
    fg = "PAHHA1M" if u in ("A", "B") else f"PAHHA1{u}"
    return {
        "flame":      f"BAB1{u}",     # 화염검출 실패
        "burner":     f"GAB1{u}",     # 버너 트러블(FG/AIR)
        "bath_all":   f"GAD1{u}",     # 수위·수온 종합이상
        "bath_ll":    f"TALLD1{u}",   # 수온 Low Low
        "bath_l":     f"TALD1{u}",    # 수온 Low
        "bath_h":     f"TAHD1{u}",    # 수온 High
        "fuel_hh":    fg,             # 연료가스압 High High (A/B 공용)
        "leak":       f"GLAB1{u}",    # 가스 누출
        "mov_torque": f"ZAF31{u}",    # 입출구 전동밸브 과토크
        "running":    f"H31{u}OH",    # 가동상태(버너 점화 사이클 — docs §8-4)
    }

# ---------------------------------------------------------------------------
# 히터별 설계 사양
# ---------------------------------------------------------------------------
# ⚠ `251016 영종설비현황.xlsx` 의 "Gcal/h" 컬럼은 **3대 모두 질량유량 오기입**이다(§11-5).
#   14.08 / 98.72 / 52 는 각각 14,200 / 99,563 / 52,444 kg/h 와 일치한다.
#   → 설계 열량은 Q = m·c_p·Δt 로 직접 산출한다. P 는 제작도서로 검증됨(768,753 kcal/h).
GAS_DENSITY_KG_NM3 = p_config.HTR_GAS_DENSITY_KG_NM3   # 0.80683
CP_GAS_J_KGK = p_config.HTR_CP_GAS_J_KGK               # 2739.8
DESIGN_DT_C = 22.4                                      # 설계 승온폭 [℃] (제작도서 P: 0→22.4)

DESIGN_FLOW_NM3_H = {"A": 17_600.0, "B": 17_600.0, "O": 123_400.0, "P": 65_000.0}
MAKER = {"A": "삼성정밀", "B": "삼성정밀", "O": "원일T&I", "P": "원일T&I"}
BATH_TYPE = {"A": "대기압식", "B": "대기압식", "O": "대기압식", "P": "진공식"}

def design_mgas_kg_s(u: str) -> float:
    return DESIGN_FLOW_NM3_H[u] * GAS_DENSITY_KG_NM3 / 3600.0

def design_duty_w(u: str) -> float:
    """Q = m·c_p·Δt. 설비현황 엑셀의 'Gcal/h' 는 쓰지 않는다(§11-5)."""
    return design_mgas_kg_s(u) * CP_GAS_J_KGK * DESIGN_DT_C

# P 전용 제작도서 확보값 (T1). A/B/O 는 1차 도서가 없어 미확보 — 추정으로만 채운다.
P_UA_W_PER_K = p_config.HTR_UA_W_PER_K            # 16,188 W/K
P_MW_CW_J_PER_K = p_config.HTR_MW_CW_J_PER_K      # 60.76 MJ/K
P_AREA_M2 = p_config.HTR_AREA_M2                  # 27.1 m²

def scaled_ua_w_per_k(u: str) -> float:
    """A/B/O 의 U·A 추정 — P 제작도서 값을 설계유량비로 스케일.

    ⚠ **T3(프록시) 등급이다.** 전열면적은 유량에 비례한다고 가정했다(같은 Δt·LMTD 설계라면
    Q ∝ m 이고 Q = U·A·ΔT_lm 이므로 U·A ∝ m). 제작사가 달라(삼성정밀 vs 원일T&I) 실제로는
    다를 수 있다. 절대값이 필요하면 해당 호기 제작도서를 확보해야 한다.
    """
    return P_UA_W_PER_K * design_mgas_kg_s(u) / design_mgas_kg_s("P")

def scaled_mw_cw_j_per_k(u: str) -> float:
    """수조 열용량 추정 — 같은 근거로 유량비 스케일(T3)."""
    return P_MW_CW_J_PER_K * design_mgas_kg_s(u) / design_mgas_kg_s("P")


# ---------------------------------------------------------------------------
# 계열(train) 구조 — 바이패스 포함 (docs §15, 2026-09-12 확정)
# ---------------------------------------------------------------------------
# 도면 004/001 로 확정된 배관: 입구헤더에서 갈라져 (a) 히터 가지배관 `250-NG-P7-302-H`~`307-H`
# (b) **히터를 건너뛰는 바이패스** `250-NG-P7-308`(보온 표기 `-H` 없음 = 냉가스) 로 나뉘고
# 출구 공통헤더에서 다시 합류한다. 바이패스 양은 변조식 밸브가 조절하며, 온도조절기가
# **헤더온도**를 설정값에 맞춘다. 개폐식 MOV 가 아니라서 DI 에 개/폐 쌍이 없고 `ZAF-31D/Q`(과토크)만 있다.
#
#   TIC-31D → ZI-31D(개도 `ZI31D`) → 헤더 31M (`TSL_TSH_TSHH-31M`)   … A/B 계열
#   TIC-31Q → ZI-31Q(개도 `ZI31Q`) → 헤더 31Z (`TSL_TSH_TSHH-31Z`)   … O/P 계열
#
# 검증(2019~2022 실측): 헤더온도가 [입구, 최고히터출구] 안에 드는 비율 98.8%/99.5%,
# 열수지로 푼 β 와 밸브개도의 Spearman +0.922/+0.896. 서로 독립인 두 계측 경로가 일치한다.

TRAIN = {"A": "M", "B": "M", "O": "Z", "P": "Z"}      # 히터 → 계열
# ⚠ 계열 M 은 도면상 히터가 **3대**다(31A·31B·31C, 배관 250-NG-P7-302~307-H 세 쌍).
#   그런데 `TI33C` 가 15년간 사실상 0 이라 C 는 관측이 없다 → 여기서는 뺄 수밖에 없다.
#   실측상 C 는 거의 돌지 않는다(A·B 가 둘 다 isolated 인 42,953행에서 헤더−입구 최대 10.19℃,
#   10℃ 초과가 0.08%). 그래도 계열 M 결과에는 "미측정 히터 1대" 위험이 남으므로
#   **합류점 식(H5) 기반 식별의 1차 대상은 계열 Z** 로 한다. docs §17-2b.
TRAIN_UNITS = {"M": ["A", "B"], "Z": ["O", "P"]}
TRAIN_INLET = {"M": "TI21Y", "Z": "TI21Z"}            # 계열 입구 공통헤더 온도
TRAIN_HEADER = {"M": "TSL_TSH_TSHH-31M", "Z": "TSL_TSH_TSHH-31Z"}   # 출구 공통헤더 온도
TRAIN_BYPASS_VALVE = {"M": "ZI31D", "Z": "ZI31Q"}     # 바이패스 조절밸브 개도 [%]

# 헤더온도 태그는 2013-06 부터 유효하다(2013년 월중 0.42 → 2014년부터 1.00). §15-4
HEADER_VALID_FROM = "2013-06-01"

# β 를 신뢰할 수 있는 조건: 히터출구와 입구의 온도차가 충분해야 나눗셈이 안 터진다.
BETA_MIN_SPREAD_C = 10.0
BETA_VALID_MIN, BETA_VALID_MAX = -0.15, 1.15          # 계측오차 여유. 밖이면 결측 처리


# ---------------------------------------------------------------------------
# 운전 상태 판정 임계값 — 설계 NTU 에서 유도 (임의값 아님)
# ---------------------------------------------------------------------------
# ε = 1 − exp(−NTU),  NTU = U·A / (m·c_p)  이므로 유량비 r = m/m_설계 에 대해
#     ε(r) = 1 − exp(−NTU_설계 / r)
# 제작도서 P: U·A = 16,188 W/K, m_설계·c_p = 14.568 × 2,739.8 = 39,913 W/K → **NTU_설계 = 0.406**
# (A/B/O 의 U·A 는 설계유량비 스케일이라 NTU_설계 가 같다 — config.scaled_ua_w_per_k 참고)
#
#   ε ≥ ε(0.25) = 0.80  ⇒ 유량이 설계의 25% 미만  ⇒ **저유량**(출구가 수조온도에 근접)
#   ε ≤ ε(3.0)  = 0.13  ⇒ 유량이 설계의 3배 초과  ⇒ 물리적으로 불가 ⇒ **차단**(사장관)
#
# ⚠ 명명 주의(2026-09-12 정정): 위 상한을 처음엔 "정체(stagnant)"라 불렀는데 **틀린 이름**이었다.
#   ε≥0.80 은 유량이 0 이라는 뜻이 아니라 설계의 25% 미만이라는 뜻이다. 설계 20% 유량이면
#   ε≈0.87 로 출구가 수조온도에 붙지만 **가스는 계속 흐르고 데워진다**. 실제로 계열 검증(T3)에서
#   두 히터가 모두 이 상태일 때도 헤더온도가 입구보다 8.8~27.7℃ 높았다 — 통가스 중이라는 증거다.
#   그래서 `lowflow` 로 바꿨다. 유량이 정말 0 인 것은 `isolated` 쪽이다.
#
# ⚠ 순환 주의: 이 임계값은 **설계** U·A 로 만든 **선별용**이다. 실제 U·A 는 이 레이어를 통과한
#   통가스 구간에서만 측정한다. 열화가 매우 심한 히터는 ε 가 낮아져 '차단'으로 오분류될 수 있으므로
#   ε(3.0) 이라는 보수적인 쪽을 썼다(설계유량의 3배는 어떤 운전에서도 나올 수 없다).
STAGNANT_FLOW_RATIO = 0.25
ISOLATED_FLOW_RATIO = 3.0

def design_ntu(u: str) -> float:
    """설계 유량에서의 NTU = U·A / (m_설계·c_p)."""
    ua = P_UA_W_PER_K if u == "P" else scaled_ua_w_per_k(u)
    return ua / (design_mgas_kg_s(u) * CP_GAS_J_KGK)

def regime_thresholds(u: str) -> tuple[float, float]:
    """(ε_차단상한, ε_정체하한). 위 유도 그대로."""
    import math
    ntu = design_ntu(u)
    return (1.0 - math.exp(-ntu / ISOLATED_FLOW_RATIO),
            1.0 - math.exp(-ntu / STAGNANT_FLOW_RATIO))
