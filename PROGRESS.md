# 진행 상황 (HTR-31P 가스히터 고장예측)

> 이 문서는 작업이 진행될 때마다 계속 업데이트한다. 최신 상태가 맨 위로 오도록 "현재 상태" 절만 덮어쓰고,
> 굵직한 마일스톤은 "타임라인"에 한 줄씩 추가한다.

## 현재 상태 (2026-09-10)

- **대상 설비**: HTR-31P(가스히터 P호기) 1대로 한정 (영종 공급관리소).
- **완료**: `prex/` 전처리 파이프라인, `trend/` EDA 시각화 파이프라인, 가스유량(`gas_flow_proxy`) 1차 역산.
- **문서**: `docs/htr31p_flow.md` — HTR-31P 가스 흐름 다이어그램 + 태그별 신뢰도(✅/⚠️/❌) 정리.
- **다음 단계**: (1) TI21Y/TI21Z 중 실제 P호기 입구가 무엇인지 확인, (2) 히터↔정압기 배관 위상을
  P&ID로 확인(가능하면 KOGAS에 PDF 요청), (3) 고장 라벨링(진짜 고장 vs. transient 판정) →
  피처 엔지니어링 → 모델링(KF-PINN 등).

## 코드 구조 (파이프라인 방식)

```
prex/    원본 CSV → 전처리 Parquet (trend/alarm/fault 이벤트)
trend/   전처리 Parquet → EDA 그림(PNG)
```

각 디렉토리는 `config.py`(설정) + 기능별 모듈 + `pipeline.py`(진입점) 구조를 공유한다.

### prex/ — 데이터 전처리 (완료)

```
.venv/bin/python -m prex.pipeline
```

- `config.py`: 대상 태그(TREND_TAGS), 알람 태그 분류(FAULT/STATUS/CONTROL), 경로.
- `trend.py`: Trend(1분 연속 시계열) 전처리 — 1분 그리드 reindex, 24시간 결측 시 segment 분리,
  PI-D2P era-wise 정규화(계기 스팬이 2015/2025년 전후로 바뀜).
- `alarm.py`: Alarm(DI/DO 이벤트) 전처리 — long 포맷 melt, 무효시각/중복 제거, 상승엣지 추출.
- `trend.py::estimate_gas_flow()`: 가스 질량유량(m_gas)이 Trend에 전량 NULL이라, 정압기(PCV-41P)
  밸브 유동식(문서 06 §3-1 G1)으로 역산 — `ZI41P`(개도)·`PI21X`(입구압)·`PI43O`(출구압, 공용헤더 가정)
  사용. **1차 근사치**(Cv/특성곡선 T3 proxy, ZI41P 스팬 휴리스틱 보정) → `docs/htr31p_flow.md` 참고.
- 출력 (`prex/output/`, git 미포함):
  - `trend_htr31p.parquet` (7,804,924행): PI21X/TI21Y/TI21Z/TI33P/TI-D2P/PI-D2P/ZI41P/PI43O/RSF41P
    + segment_id + gas_flow_proxy/ZI41P_frac/control_error_p(파생)
  - `alarm_events_htr31p.parquet` (95,770행): 정제된 전체 DI/DO 이벤트
  - `fault_events_htr31p.parquet` (43,206행): 상승엣지(발생 시점)만 추출

### trend/ — EDA 시각화 (신규, 완료)

```
.venv/bin/python -m trend.pipeline
```

prex 출력을 입력으로 받아 일반적인 데이터 탐색 그림(상관관계/히트맵/박스플롯 등)을 생성한다.

- `config.py`: prex.config 재사용(태그 메타 중복 정의 안 함), 그림 스타일/샘플링 크기 설정.
- `style.py`: seaborn 테마 + 한글 폰트(Noto Sans CJK KR) 적용.
- `data.py`: parquet 로더 3종.
- `plots/overview.py`: 태그별 일평균 전체 기간 추이(line).
- `plots/distribution.py`: 히스토그램, 박스플롯, PI-D2P era별 박스플롯(원본 vs 정규화 비교).
- `plots/correlation.py`: 상관계수 히트맵, 산점도 행렬(pairplot, 5,000행 샘플링).
- `plots/missingness.py`: 태그별 월간 데이터 커버리지 히트맵.
- `plots/alarms.py`: 태그별 이벤트 발생 건수(symlog bar), 분류별 비중, FAULT 월별 발생 추이.
- 출력 (`trend/output/figures/`, git 미포함): `01_~10_*.png` 10종.

**의존성 추가**: matplotlib, seaborn을 `.venv`에 새로 설치. `requirements.txt`에 기록해 둠
(이전에는 pandas/numpy/pyarrow만 있었음).

## EDA 결과에서 눈에 띈 점 (다음 단계 참고용)

- STATUS 태그(특히 `H31POH`, 가동상태 토글)가 FAULT 태그보다 발생 건수가 수십~수천 배 많아
  단순 빈도 기준으로 보면 안 됨 (→ 알람 카운트 그림은 symlog 스케일 사용).
- FAULT 이벤트는 2014~2018년 사이에 집중 발생(특히 2015년 초 스파이크)하고 이후 낮은 수준 유지.
  라벨링 단계에서 이 시기 데이터 품질/원인을 별도로 확인할 필요.
- TI21Y/TI21Z(히터 입구온도)는 상관계수 0.87로 거의 중복 정보 — 모델 피처 선택 시 고려.
- PI-D2P는 era(2011-2014 / 2015-2024 / 2025-)별로 값 분포가 확연히 다름 → `PI-D2P_norm`
  (era-wise z-score) 컬럼을 원본 대신 사용 권장.
- 데이터 커버리지는 대체로 양호하나 2018-10, 2023-02~09, 2024-05~08, 2026-05~ 부근에 결측 구간 존재.

## 타임라인

- 2026-09-09: `prex/` 전처리 파이프라인 완료(HTR-31P 1대 한정 결정 포함). `trend/` EDA 시각화
  파이프라인 신규 작성 및 검증 완료.
- 2026-09-10: 6개 태그만으로 고장예측이 충분한지 검토 → 가스 질량유량(m_gas) 완전 결측 확인.
  정압기 밸브식으로 `gas_flow_proxy` 역산해 `prex/`에 추가. HTR-31P 가스 흐름/태그 신뢰도를
  정리한 `docs/htr31p_flow.md` 작성 — TI21Y/TI21Z 입구 태그 모호성, 히터↔정압기 배관 위상
  미확정(P&ID 필요), Q_burner 실측 없음을 미해결 항목으로 명시.
