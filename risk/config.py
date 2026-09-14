"""빙결 위험도 예측 설정.

목표: **계량설비 출구 가스온도가 물리적 빙결점 0℃ 아래로 내려갈 위험**을 확률로 낸다.

왜 이 목표인가 (docs/07 의 전수 검정 뒤에 남은 유일한 길):
  · 고장 라벨이 필요 없다 — 임계 초과가 데이터에서 직접 관측된다
  · 유량·`U·A`·`fire` 가 필요 없다 — 전부 산출 불가로 확정된 것들
  · 물리 근거가 분명하다 — 히터의 존재 이유가 정압기 줄-톰슨 냉각(−0.34~0.41 ℃/bar) 상쇄다
  · 검증이 된다 — 캘리브레이션(예측확률 vs 실제 발생률), 라벨 없이도 측정 가능

⚠ **사건이 2016년 이후 사라졌다.** 0℃ 미만 발생률이 M계열 2011~2014 11~16% → 2018~2026 0.0~0.2%.
   그래서 **이진 사건이 아니라 '여유(margin)' 를 연속값으로 예측**한다:
       margin(t, H) = min(T61 over (t, t+H]) − 0℃
   여유는 사건이 0건인 해에도 정의되고 변동한다. 위험확률은 `P(margin < 0)` 으로 낸다.
"""
from __future__ import annotations

from pathlib import Path

from prex import config as p_config
from prex_multi import config as m_config

PROJECT_ROOT = p_config.PROJECT_ROOT
TREND_PARQUET = m_config.OUTPUT_DIR / "trend_all.parquet"
RAW_TREND_DIR = p_config.TREND_DIR
OUTPUT_DIR = PROJECT_ROOT / "risk" / "output"

FREEZE_C = 0.0                      # 물리적 빙결점
HORIZONS_H = [6, 24]                # 예측 지평
FREQ = "1h"                         # 집계 주기 (1분 → 1시간)

# 계열: 출구온도 태그와 그 계열의 히터
TRAINS = {
    "M": {"t61": "TI61M", "p61": "PI61M", "hdr": "TSL_TSH_TSHH-31M",
          "t_in": "TI21Y", "units": ["A", "B"]},
    "Z": {"t61": "TI61Z", "p61": "PI61Z", "hdr": "TSL_TSH_TSHH-31Z",
          "t_in": "TI21Z", "units": ["O", "P"]},
}
P_IN = "PI21X"

# 시간 분할 — 누수 방지. 검증/시험은 미래 구간으로만.
TRAIN_END = "2019-01-01"
VALID_END = "2022-01-01"
PURGE_H = 24                        # 분할 경계에서 지평만큼 버린다

RANDOM_SEED = 42

# --- 2026-09-13 신설
HEATFLOW_PARQUET = m_config.OUTPUT_DIR / "heatflow_cycles.parquet"
USE_HEATFLOW = False       # True 면 수조 열수지 유량을 특징으로 쓴다 (문서 12). 문서 08 재현을 위해 기본 끔
REGIME_FROM = None         # 학습 시작 시점 제한. 2015~2016 운전 체제 변경(문서 15) 대응.
                           # None 이면 전 기간. "2016-07-01" 이면 새 체제만.
