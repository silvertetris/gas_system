"""HTR-31P EDA 시각화 파이프라인 실행 진입점.

사용법:
    .venv/bin/python -m trend.pipeline

입력: prex/output/*.parquet (먼저 `python -m prex.pipeline` 으로 생성되어 있어야 함)
출력 (PNG, trend/output/figures/):
    01_time_series_overview.png   태그별 일평균 전체 기간 추이
    02_histograms.png             태그별 분포 히스토그램
    03_boxplots.png               태그별 박스플롯
    04_pi_d2p_era_boxplot.png     PI-D2P 계기 스팬 era별 박스플롯(원본/정규화 비교)
    05_correlation_heatmap.png    태그 간 상관계수 히트맵
    06_pairplot.png               태그 산점도 행렬(샘플링)
    07_missing_heatmap.png        태그별 월간 데이터 커버리지(결측) 히트맵
    08_alarm_counts_by_tag.png    태그별 이벤트 발생 건수
    09_alarm_category_share.png   분류별 이벤트 발생 건수
    10_alarm_monthly_trend.png    FAULT 이벤트 월별 발생 추이
"""
from __future__ import annotations

import logging

from . import config, data
from .plots import alarms, correlation, distribution, missingness, overview


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=== 데이터 로드 ===")
    trend = data.load_trend()
    fault_events = data.load_fault_events()

    logging.info("=== Trend 개요/분포 그림 ===")
    overview.plot_time_series_overview(trend)
    distribution.plot_histograms(trend)
    distribution.plot_boxplots(trend)
    distribution.plot_pi_d2p_by_era(trend)

    logging.info("=== 상관관계 그림 ===")
    correlation.plot_correlation_heatmap(trend)
    correlation.plot_pairplot(trend)

    logging.info("=== 결측/커버리지 그림 ===")
    missingness.plot_missing_heatmap(trend)

    logging.info("=== 알람/이벤트 그림 ===")
    alarms.plot_alarm_counts_by_tag(fault_events)
    alarms.plot_alarm_category_share(fault_events)
    alarms.plot_alarm_monthly_trend(fault_events)

    logging.info("=== 완료: %s ===", config.FIGURE_DIR)


if __name__ == "__main__":
    main()
