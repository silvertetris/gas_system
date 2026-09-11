"""HTR-31P Trend(1분 연속 시계열) 전처리.

원본: data/Trend 데이터/AI_CV/*.csv (월별 175개 파일)
체크리스트 출처: data/docs/03_시계열데이터_구조와품질.md §8
  1. Time + 대상 태그만 읽기
  2. Time → datetime, NaT 제거
  3. Time 기준 중복 제거 (keep='last')
  4. 1분 그리드로 reindex → 결측은 명시적 NaN (보간 금지)
  5. 24시간 이상 결측은 학습 시퀀스 경계(segment)로 표시
  6. (해당 없음 — TI33C 등 죽은 태그는 애초에 TREND_TAGS에 없음)
  7. PI-D2P는 era-wise 재정규화
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


_ALL_INPUT_TAGS = config.TREND_TAGS + config.FLOW_INPUT_TAGS


def _read_one_month(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["Time", *_ALL_INPUT_TAGS],
        dtype={tag: "float64" for tag in _ALL_INPUT_TAGS},
    )
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df.dropna(subset=["Time"])


def load_trend_raw() -> pd.DataFrame:
    """175개 월별 CSV를 읽어 하나로 합친다 (Time 파싱·NaT 제거까지)."""
    paths = sorted(config.TREND_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Trend CSV를 찾을 수 없습니다: {config.TREND_DIR}")
    frames = [_read_one_month(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    logger.info("Trend raw 로드: 파일 %d개, %d행", len(paths), len(df))
    return df


def dedup_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """같은 Time 중복 시 마지막 값 유지 (2019년 이후 주 단위 재기록 중복 대응)."""
    before = len(df)
    df = df.sort_values("Time").drop_duplicates(subset="Time", keep="last")
    logger.info("Time 중복 제거: %d -> %d행", before, len(df))
    return df


def compute_segments(observed_time: pd.Series, gap_hours: float = config.SEGMENT_GAP_HOURS) -> pd.Series:
    """실측 시각들 사이 gap_hours 이상 결측이면 새 segment로 취급.

    Returns: 실측 시각을 index로 하는 segment_id Series (0부터 시작).
    """
    t = observed_time.sort_values()
    boundary = t.diff() >= pd.Timedelta(hours=gap_hours)
    segment_id = boundary.cumsum()
    return pd.Series(segment_id.values, index=t.values, name="segment_id")


def build_minute_grid(df: pd.DataFrame) -> pd.DataFrame:
    """1분 간격 전체 그리드로 reindex. 결측은 보간하지 않고 명시적 NaN으로 남긴다."""
    seg = compute_segments(df["Time"])
    df = df.set_index("Time")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    grid = df.reindex(full_index)
    grid.index.name = "Time"

    # 그리드의 각 시각에, 그 시각 이전(또는 같은 시각)의 가장 최근 실측 경계를 전파한다.
    grid["segment_id"] = seg.reindex(grid.index, method="ffill")

    logger.info(
        "1분 그리드 생성: %d행 (%s ~ %s), segment %d개",
        len(grid), full_index.min(), full_index.max(), grid["segment_id"].nunique(),
    )
    return grid


def add_era_normalization(df: pd.DataFrame, column: str = "PI-D2P") -> pd.DataFrame:
    """계기 스팬이 시기별로 바뀐 태그에 대해 구간별(era) z-score 정규화 컬럼을 추가한다."""
    era = pd.Series(pd.NA, index=df.index, dtype="object")
    for label, start, end in config.PI_D2P_ERAS:
        mask = df.index >= pd.Timestamp(start)
        if end is not None:
            mask &= df.index < pd.Timestamp(end)
        era.loc[mask] = label
    df[f"{column}_era"] = era

    normed = pd.Series(np.nan, index=df.index, dtype="float64")
    for label in era.dropna().unique():
        mask = era == label
        vals = df.loc[mask, column]
        mu, sigma = vals.mean(), vals.std()
        if sigma and not np.isnan(sigma):
            normed.loc[mask] = (vals - mu) / sigma
    df[f"{column}_norm"] = normed
    return df


def estimate_gas_flow(df: pd.DataFrame) -> pd.DataFrame:
    """정압기(PCV-41P) 밸브 유동식으로 히터 통과 가스유량을 역산한다 (1차 근사치).

    문서 06 §3-1 G1식: Q = Cv · f(z) · sqrt((P_in^2 - P_out^2) / (G·T))
    Trend에 유량계(FI61x)가 전량 NULL이라 이 역산이 사실상 유일한 대안이다 (문서 04 §5-1).

    ⚠⚠ 이 값은 "HTR-31P 통과유량"이 아니다 (2026-09-11 P&ID 확인, docs/htr31p_flow.md §7-2).
    히터 2대(31O·31P) 출구가 공통 헤더 400-NG-P7-358로 합류한 뒤 정압기 4라인(41O/P/Q/R)으로
    분기하므로 PCV-41P 통과유량 ≠ HTR-31P 통과유량이다. gas_flow_proxy는 **"공통헤더에서 41P
    라인이 뽑아 간 유량"**으로만 해석할 것. H2식의 m_gas 대용으로 쓰면 편향된다 —
    m_gas는 KF-PINN 잠재변수로 공동추정하는 쪽이 맞다 (config.FLOW_INPUT_TAGS 주석 참조).

    ⚠ config.GAS_SG·PCV41P_CV는 P호기 자체가 아닌 남사 문서의 동일 모델 카탈로그 값(T3 proxy)이고,
    f(z)는 실제 특성곡선 대신 선형 근사다. 절대량이 아니라 상대적 추이(gas_flow_proxy)로만 쓸 것 —
    Cv/특성곡선을 영종 P호기 정압기 도서로 확인하기 전까지 절대치를 신뢰하지 말 것.
    """
    z_raw = df["ZI41P"]
    z_pct = z_raw.where(z_raw <= config.ZI41P_RESCALE_ABOVE, z_raw / 5.0).clip(upper=100.0)
    z_frac = (z_pct / 100.0).clip(lower=0.0)

    p_in = df["PI21X"]
    p_out = df["PI43O"]
    # 유동식의 T는 정압기를 지나는 가스의 절대온도. 히터 입구온도로 근사하되
    # TI21Y(=A/B 도시가스 계열)는 섞지 않는다 — config.HTR31P_INLET_TAG 주석 참조.
    t_kelvin = df[config.HTR31P_INLET_TAG] + 273.15

    dp_sq = (p_in**2 - p_out**2).clip(lower=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_proxy = config.PCV41P_CV * z_frac * np.sqrt(dp_sq / (config.GAS_SG * t_kelvin))

    df["ZI41P_frac"] = z_frac
    df["gas_flow_proxy"] = q_proxy
    df["control_error_p"] = df["PI43O"] - df["RSF41P"]  # 문서 05 §4-3: 열화지표 후보(e = PV - SP)
    return df


def add_vacuum_regime(df: pd.DataFrame) -> pd.DataFrame:
    """진공 상태 regime 라벨을 붙인다 (`deep` / `shallow`).

    (2026-09-11 신설 — docs/htr31p_flow.md §8)
    `PI-D2P_era`가 "계기 스팬이 바뀐 구간"이라는 **가정**에서 나온 달력 기반 라벨인 반면,
    이건 실제 관측값으로 판정한다: 진공압 월 중앙값이 config.VACUUM_DEEP_THRESHOLD 보다
    깊으면 `deep`, 아니면 `shallow`. 두 라벨이 어긋나는 시각이 있으면 그 자체가 점검 대상이다.

    ⚠ 이 라벨은 **"PI-D2P 수준 구간"** 이지 "진공 상실/정상"이 아니다.
    버너 ON 구간만 비교하면 승온폭이 deep 20.25℃ vs shallow 20.47℃로 사실상 같다 —
    즉 이 구간 차이는 히터 성능과 연동되지 않는다(docs/htr31p_flow.md §8-3).
    계기 스팬 변경일 가능성이 남아 있으므로 열화 지표로 직접 쓰지 말 것.

    ⚠ 월 중앙값으로 판정하므로 `PI-D2P_era`(일 단위 경계)와 **2014-10-21~10-31 약 11일
    (15,809행)이 어긋난다**. 2014년 10월은 21일까지가 deep 이라 월 중앙값이 deep 으로 찍히기
    때문이다. 알려진 해상도 차이이고 버그가 아니다 — 정확한 경계가 필요하면 `PI-D2P_era`를 쓸 것.
    """
    monthly = df["PI-D2P"].resample("MS").median()
    label = np.where(monthly < config.VACUUM_DEEP_THRESHOLD, "deep", "shallow")
    regime = pd.Series(label, index=monthly.index).reindex(df.index, method="ffill")
    df["vacuum_regime"] = regime.where(df["PI-D2P"].notna())
    counts = df["vacuum_regime"].value_counts(dropna=True)
    logger.info("진공 regime: %s", ", ".join(f"{k} {v:,}행" for k, v in counts.items()))
    return df


def add_heat_exchange(df: pd.DataFrame) -> pd.DataFrame:
    """열교환기 효율 ε 과 NTU 를 계산한다 — **신뢰 태그 3개만 쓰는 무가정 지표**.

    (docs/kfpinn_state_space.md §2. 2026-09-11 신설)

    튜브번들을 "수조온도 T_bath 로 유지되는 열원에 대한 열교환기"로 보면 ε-NTU 관계가 성립한다:

        ε   = (T_out − T_in) / (T_bath − T_in)        ← TI33P, TI21Z, TI-D2P 만으로 계산
        NTU = −ln(1 − ε) = U·A / (m_gas · c_p)

    이 두 줄이 중요한 이유:
      - **`gas_flow_proxy`와 달리 `PI43O`·`ZI41P`·`Cv`·밸브특성곡선이 하나도 안 들어간다.**
        P&ID로 흔들린 재료(§7-2·§7-3)를 전부 우회하고, T1 등급 태그 3개만 쓴다.
      - ε 는 매 분 직접 관측되고, NTU 는 곧 **`U·A`와 `m_gas`의 비(比)** 다.
        즉 이 한 식으로는 둘을 분리할 수 없고(구조적 미식별), 분리하려면 H1(수조 열수지)이
        추가로 필요하다 — 그래서 `m_gas`를 KF-PINN **잠재변수**로 두는 설계가 나온다.

    유효조건: 구동 온도차 `T_bath − T_in` 이 충분히 커야(EPS_MIN_DRIVE_C) 비가 안정적이다.
    범위 밖(계기 dropout·역전 등)은 NaN 으로 남긴다 — 보간하지 않는다.
    """
    t_in = df[config.HTR31P_INLET_TAG]
    drive = df["TI-D2P"] - t_in                       # 구동 온도차 (수조 − 가스입구)
    eps = (df["TI33P"] - t_in) / drive.where(drive > config.EPS_MIN_DRIVE_C)
    eps = eps.where(eps.between(config.EPS_VALID_MIN, config.EPS_VALID_MAX))

    df["htx_drive_c"] = drive
    df["htx_eps"] = eps
    with np.errstate(invalid="ignore", divide="ignore"):
        df["htx_ntu"] = -np.log(1.0 - eps)             # = U·A / (m_gas·c_p)
    n = int(eps.notna().sum())
    logger.info("ε/NTU 산출: 유효 %d행 (%.1f%%), ε 중앙값 %.3f",
                n, 100 * n / max(len(df), 1), float(eps.median()) if n else float("nan"))
    return df


def run_trend_pipeline() -> pd.DataFrame:
    """Trend 전처리 전체 실행: 로드 → 중복제거 → 1분 그리드 → era 정규화 → 유량 역산 → 진공 regime → ε/NTU."""
    df = load_trend_raw()
    df = dedup_by_time(df)
    grid = build_minute_grid(df)
    grid = add_era_normalization(grid, column="PI-D2P")
    grid = estimate_gas_flow(grid)
    grid = add_vacuum_regime(grid)
    grid = add_heat_exchange(grid)
    return grid
