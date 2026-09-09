"""HTR-31P Alarm(DI/DO 이벤트 로그) 전처리.

원본:
  data/Alram 데이터/DI(파일이 커서 태그 50개 단위로 분리함)/*.csv (873태그, 50개씩 분할)
  data/Alram 데이터/DataExtract_..._DO.csv (258태그)

체크리스트 출처: data/docs/03_시계열데이터_구조와품질.md §8
  1. 대상 태그를 long 포맷으로 melt
  2. Time에 'n/a'가 포함된 행 제거 (일별 요약행 — 전체의 45%)
  3. 2011~2026 범위 밖 제거
  4. (tag, Time, val) 중복 제거 (KPOS 일별 리포트 이어붙이기로 인한 대량 중복)
  5. val==1 상승엣지만 추출 → 고장/이벤트 테이블
  6. 태그별 설명·분류(FAULT/STATUS/CONTROL) 부여
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _read_header(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.readline().rstrip("\n").split(",")


def find_tag_files(tags: Iterable[str], search_dirs: Iterable[Path]) -> dict[Path, list[str]]:
    """주어진 태그들이 어느 CSV 파일(들)에 흩어져 있는지 헤더만 스캔해 찾는다.

    DI 태그는 50개씩 18개 파일로 쪼개져 있어, 필요한 파일만 골라 usecols로 읽기 위함.
    """
    remaining = set(tags)
    found: dict[Path, list[str]] = {}
    for d in search_dirs:
        for path in sorted(d.glob("*.csv")):
            header = _read_header(path)
            hit = [t for t in header if t in remaining]
            if hit:
                found[path] = hit
                remaining -= set(hit)
        if not remaining:
            break
    if remaining:
        logger.warning("찾지 못한 태그: %s", sorted(remaining))
    return found


def load_wide(path: Path, tags: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["Time", *tags], dtype={t: "float64" for t in tags})


def melt_to_long(df: pd.DataFrame, tags: list[str], source: str) -> pd.DataFrame:
    """행마다 값이 채워진 컬럼이 하나뿐인 wide 포맷 → (Time, tag, value) long 포맷."""
    long = df.melt(id_vars="Time", value_vars=tags, var_name="tag", value_name="value")
    long = long.dropna(subset=["value"])
    long["source"] = source
    return long


def load_all_events(tag_dir_map: dict[str, tuple[dict[str, str], list[Path]]]) -> pd.DataFrame:
    """{source: (tags, search_dirs)} 를 받아 필요한 파일만 찾아 읽고 long 포맷으로 합친다."""
    frames = []
    for source, (tags, dirs) in tag_dir_map.items():
        file_map = find_tag_files(tags.keys(), dirs)
        for path, hit_tags in file_map.items():
            wide = load_wide(path, hit_tags)
            frames.append(melt_to_long(wide, hit_tags, source=source))
    if not frames:
        raise RuntimeError("어떤 alarm 태그도 찾지 못했습니다.")
    return pd.concat(frames, ignore_index=True)


def drop_invalid_time(long: pd.DataFrame) -> pd.DataFrame:
    """'YYYY-MM-DD n/a' 같은 시각 없는 행, 파싱 실패, 범위 밖 연도를 제거한다."""
    before = len(long)
    long = long[~long["Time"].astype(str).str.contains("n/a")]
    long = long.assign(Time=pd.to_datetime(long["Time"], errors="coerce"))
    long = long.dropna(subset=["Time"])
    long = long[(long["Time"] >= config.ALARM_TIME_MIN) & (long["Time"] <= config.ALARM_TIME_MAX)]
    logger.info("무효 시각 제거: %d -> %d행", before, len(long))
    return long


def dedup_events(long: pd.DataFrame) -> pd.DataFrame:
    """(tag, Time, value) 중복 제거 — KPOS가 일별 알람 리포트를 이어붙여 내보내는 문제 대응."""
    before = len(long)
    long = long.drop_duplicates(subset=["tag", "Time", "value"])
    long = long.sort_values(["tag", "Time"]).reset_index(drop=True)
    logger.info("(tag,Time,value) 중복 제거: %d -> %d행", before, len(long))
    return long


def annotate(long: pd.DataFrame) -> pd.DataFrame:
    """태그별 설명·분류(FAULT/STATUS/CONTROL)를 조인한다."""
    meta = (
        pd.DataFrame.from_dict(config.ALARM_TAG_META, orient="index")
        .rename_axis("tag")
        .reset_index()
    )
    return long.merge(meta, on="tag", how="left")


def extract_rising_edges(long: pd.DataFrame) -> pd.DataFrame:
    """태그별 val 0(또는 무)->1 상승엣지만 추출 = 알람/이벤트 '발생' 시점.

    문서 04 §4-1: 대부분 SET 후 수 초~수십 초 내 자동 RESET되는 순간 이상(transient)이므로,
    지속시간·반복횟수 기반 '진짜 고장' 판정은 이 결과를 입력으로 다음 단계(라벨링)에서 수행한다.
    """
    edges = []
    for tag, g in long.groupby("tag", sort=False):
        g = g.sort_values("Time")
        prev = g["value"].shift(1)
        rising = g[(g["value"] == 1) & (prev != 1)]
        edges.append(rising)
    if not edges:
        return long.iloc[0:0]
    out = pd.concat(edges, ignore_index=True)
    return out.sort_values("Time").reset_index(drop=True)


def run_alarm_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alarm 전처리 전체 실행. Returns (정제된 전체 이벤트 long표, 상승엣지만 뽑은 이벤트표)."""
    tag_dir_map = {
        "DI": (config.DI_TAGS, [config.ALARM_DI_DIR]),
        "DO": (config.DO_TAGS, [config.ALARM_DIR]),
    }
    long = load_all_events(tag_dir_map)
    long = drop_invalid_time(long)
    long = dedup_events(long)
    long = annotate(long)
    events = extract_rising_edges(long)
    return long, events
