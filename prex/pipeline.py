"""HTR-31P 가스히터 고장예측용 데이터 전처리 파이프라인 실행 진입점.

사용법:
    .venv/bin/python -m prex.pipeline

출력 (Parquet, config.OUTPUT_DIR):
    trend_htr31p.parquet        1분 그리드 연속 시계열 (PI21X/TI21Y/TI21Z/TI33P/TI-D2P/PI-D2P + segment_id)
    alarm_events_htr31p.parquet 정제된 DI/DO 이벤트 long표 (중복 제거 후 전체, SET/RESET 모두 포함)
    fault_events_htr31p.parquet 상승엣지(발생 시점)만 추출한 이벤트표 — 라벨링 다음 단계의 입력
"""
from __future__ import annotations

import logging

from . import config
from .alarm import run_alarm_pipeline
from .trend import run_trend_pipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=== Trend 파이프라인 시작 ===")
    trend = run_trend_pipeline()
    trend_path = config.OUTPUT_DIR / "trend_htr31p.parquet"
    trend.reset_index().to_parquet(trend_path, index=False)
    logging.info("Trend 저장 완료: %s (%d행)", trend_path, len(trend))

    logging.info("=== Alarm 파이프라인 시작 ===")
    events_long, fault_events = run_alarm_pipeline()
    events_path = config.OUTPUT_DIR / "alarm_events_htr31p.parquet"
    fault_path = config.OUTPUT_DIR / "fault_events_htr31p.parquet"
    events_long.to_parquet(events_path, index=False)
    fault_events.to_parquet(fault_path, index=False)
    logging.info("Alarm 전체 이벤트 저장 완료: %s (%d행)", events_path, len(events_long))
    logging.info("상승엣지 이벤트 저장 완료: %s (%d행)", fault_path, len(fault_events))


if __name__ == "__main__":
    main()
