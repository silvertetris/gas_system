"""4대 히터 Trend 전처리 — 계열 바이패스 구조를 반영한 관측 레이어 (docs §15).

⚠ 누수·결측 처리 원칙은 `prex/` 와 동일하다: 1분 그리드 reindex, **보간 금지**,
   24시간 이상 결측은 segment 경계.

2026-09-12 개편 — 무엇이 바뀌었나:
  이전 버전은 히터 3태그(`TI21x`/`TI33x`/`TI-D2x`)만으로 ε·NTU 를 계산했다. 그런데 실제 배관은
  **히터와 나란히 냉가스 바이패스가 있고**, 온도조절기가 그 양을 조절해 출구 공통헤더 온도를
  맞춘다(§15). 그래서
    (a) 가스의 34~59%(중앙값)는 애초에 히터를 통과하지 않고,
    (b) 히터로 가는 유량이 0 에 가까운 시간대의 ε 을 열전달로 읽으면 전부 오독이다.
  이 파일은 그 두 가지를 관측 가능한 양으로 분리한다:
    - **β** (바이패스 분율)  : 합류점 열수지에서 직접 계산
    - **regime** (운전 상태) : 통가스 / 정체 / 차단 — 설계 NTU 에서 유도한 임계값으로 판정
  ε·NTU 는 그대로 두되 **`regime == "flow"` 인 구간에서만 해석**해야 한다.
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd

from prex import config as p_config
from prex import trend as p_trend

from . import config

logger = logging.getLogger(__name__)

REGIMES = ["flow", "lowflow", "isolated"]


def all_tags() -> list[str]:
    """4대 히터 + 계열(바이패스·헤더)이 쓰는 Trend 태그 전체."""
    tags = set()
    for u in config.UNITS:
        tags.update(t for t in config.trend_tags(u).values() if t)
    for tr in config.TRAIN_UNITS:
        tags.add(config.TRAIN_INLET[tr])
        tags.add(config.TRAIN_HEADER[tr])
        tags.add(config.TRAIN_BYPASS_VALVE[tr])
    return sorted(tags)


def load_raw() -> pd.DataFrame:
    paths = sorted(config.TREND_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Trend CSV 없음: {config.TREND_DIR}")
    tags = all_tags()
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=["Time", *tags], dtype={t: "float64" for t in tags})
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        frames.append(df.dropna(subset=["Time"]))
    df = pd.concat(frames, ignore_index=True)
    logger.info("Trend raw: 파일 %d개, %d행, 태그 %d개", len(paths), len(df), len(tags))
    return df


def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = p_trend.dedup_by_time(df)
    return p_trend.build_minute_grid(df)


# ---------------------------------------------------------------------------
# 버너 가동상태 — DI 로그의 3가지 함정을 여기서 한 번에 처리한다
# ---------------------------------------------------------------------------
# `H31xOH` 는 히터 x 의 버너 점화 사이클이다(§8-4: "가동 기간"이 아니라 점화 ON/OFF, 중앙 15분).
# **원본 DI CSV 를 그대로 읽으면 반드시 틀린다.** 실측한 세 가지 함정(§14-4):
#
#   1. 행 중복 93%      — `H31POH` 시각 있는 이벤트 1,112,185건 → (Time,값) 중복 제거 시 78,791건.
#                         제거 안 하면 "연속 이벤트의 값이 바뀌는 비율"이 0.084 로 나와 상태 로그로 안 보인다.
#   2. 시각 없음 47.5%  — `Time` 이 `2011-10-06 n/a` 처럼 **날짜만** 있는 행. `to_datetime(errors=coerce)`
#                         가 NaT 로 만들어 조용히 사라진다. 이벤트가 있는 4,781일 중 435일(9.1%)은
#                         시각 있는 이벤트가 하나도 없어 **그날 버너 상태를 모른다** → NaN 으로 남긴다.
#   3. 값 비트팩 6%     — `4294967041`, `87565569` 같은 값이 섞인다. **하위 1비트가 실제 상태**다
#                         (`4294967041 & 1 = 1`). `v & 1` 로 복원한다.
#
# 검증(§14-4): 이렇게 만든 상태로 수조온도 변화율을 보면 ON 중앙 +0.488 ℃/h, OFF 중앙 −0.976 ℃/h,
# OFF 구간의 90.8% 가 실제 냉각이다. 방향이 맞는다.
#
# ⚠ 이 로직을 분석 모듈에서 다시 구현하지 말 것. 여기서 만든 `burner_{u}` 컬럼을 쓴다.

BURNER_DI_FILES = {"DI_06": ["H31AOH", "H31BOH", "H31OOH"], "DI_07": ["H31POH"]}


def add_burner_state(grid: pd.DataFrame) -> pd.DataFrame:
    """히터별 버너 점화상태를 `burner_A/B/O/P` 컬럼으로 붙인다 (0/1, 미상은 NaN)."""
    for fkey, tags in BURNER_DI_FILES.items():
        matches = [p for p in glob.glob(str(config.ALARM_DI_DIR / "*.csv")) if fkey in p]
        if not matches:
            raise FileNotFoundError(f"DI 파일 없음: {fkey}")
        raw = pd.read_csv(matches[0], usecols=["Time"] + tags, dtype=str)
        for t in tags:
            u = t[3]
            x = raw[["Time", t]].dropna()
            x = x[~x["Time"].str.contains("n/a")]                      # 함정 2
            v = pd.to_numeric(x[t], errors="coerce").astype("Int64") & 1   # 함정 3
            x = x.assign(v=v).dropna(subset=["v"])
            x["Time"] = pd.to_datetime(x["Time"])
            x = x.drop_duplicates(["Time", "v"]).sort_values("Time")   # 함정 1
            x = x[x["v"].diff() != 0]                                  # 실제 상태변화만
            # DI 는 변화 시점만 기록되는 이벤트 로그 → 직전 상태로 채운다.
            m = pd.merge_asof(pd.DataFrame({"Time": grid.index}), x[["Time", "v"]],
                              on="Time", direction="backward")
            grid[f"burner_{u}"] = pd.Series(m["v"].to_numpy(), index=grid.index).astype("Float64")
            on = grid[f"burner_{u}"]
            logger.info("  [%s] 버너 상태변화 %d건 → ON 비율 %.1f%%, 미상 %.2f%%",
                        u, len(x), 100 * float((on == 1).mean()), 100 * float(on.isna().mean()))
    return grid


# ---------------------------------------------------------------------------
# 계열 단위 — 바이패스 분율 β
# ---------------------------------------------------------------------------
def add_train_features(grid: pd.DataFrame) -> pd.DataFrame:
    """계열별 바이패스 분율 `beta_M` / `beta_Z` 와 보조 컬럼.

    합류점 열수지 (c_p 는 양변에서 소거된다):
        T_header = (1−β)·T_히터출구 + β·T_입구      →      β = (T_out − T_header)/(T_out − T_in)

    `T_히터출구` 는 그 계열 히터들 중 **최고 출구온도**를 쓴다. 두 히터가 동시에 다른 온도로
    통가스 중이면 실제로는 유량가중 평균이라 β 가 살짝 과대평가되지만, 한쪽이 정체·차단인
    경우(대부분)에는 최고값이 곧 통가스 히터의 출구라 정확하다. 개별 유량 분배 f_A/f_B 를
    푸는 것은 이 레이어의 일이 아니다(설계 U·A 가 필요 — 다음 단계).
    """
    for tr, units in config.TRAIN_UNITS.items():
        t_in = grid[config.TRAIN_INLET[tr]]
        t_hdr = grid[config.TRAIN_HEADER[tr]]
        t_hot = grid[[f"TI33{u}" for u in units]].max(axis=1)

        # 헤더온도 태그는 2013-06 이전엔 상수 0 이다(§15-4). 그 구간은 β 를 만들지 않는다.
        valid = grid.index >= pd.Timestamp(config.HEADER_VALID_FROM)
        spread = t_hot - t_in
        ok = valid & (spread > config.BETA_MIN_SPREAD_C) & (t_hdr > 0)

        beta = ((t_hot - t_hdr) / spread.where(ok))
        beta = beta.where(beta.between(config.BETA_VALID_MIN, config.BETA_VALID_MAX))

        grid[f"t_hdr_{tr}"] = t_hdr.where(valid & (t_hdr > 0))
        grid[f"t_hot_{tr}"] = t_hot
        grid[f"beta_{tr}"] = beta
        grid[f"valve_{tr}"] = grid[config.TRAIN_BYPASS_VALVE[tr]]

        n = int(beta.notna().sum())
        logger.info("  [계열 %s] β 유효 %d행(%.1f%%), 중앙 %.3f, p10~p90 %.3f~%.3f",
                    tr, n, 100 * n / len(grid),
                    float(beta.median()) if n else np.nan,
                    float(beta.quantile(0.10)) if n else np.nan,
                    float(beta.quantile(0.90)) if n else np.nan)
    return grid


# ---------------------------------------------------------------------------
# 히터 단위 — ε·NTU 와 운전 상태
# ---------------------------------------------------------------------------
def add_unit_features(grid: pd.DataFrame) -> pd.DataFrame:
    """히터별 `dT_u`, `drive_u`, `eps_u`, `ntu_u`, `regime_u` 를 붙인다.

    `regime_u` ∈ {flow, stagnant, isolated, NaN}:
      - **isolated** ε ≤ ε(3.0×설계유량): 뜨거운 수조를 두고 출구가 입구에 붙어 있다.
        설계유량의 3배는 물리적으로 불가하므로 "가스가 이 히터로 가지 않는다"로 읽는다.
      - **lowflow**  ε ≥ ε(0.25×설계유량): 유량이 설계의 25% 미만이라 출구가 수조온도에 근접.
        **유량이 0 이라는 뜻이 아니다** — 계속 흐르며 데워진다. 다만 ε→1 이라 NTU 가 발산해
        U·A 추정 정밀도가 없다(1−ε 의 로그라 분해능 0.061℃가 그대로 증폭된다).
      - **flow**     그 사이. **열전달 해석은 이 구간에서만 유효하다.**
      - NaN          구동온도차가 작아(ε 분모) 판정 불가.
    임계값 유도는 config.regime_thresholds 주석 참고 — 임의 상수가 아니다.
    """
    for u in config.UNITS:
        t = config.trend_tags(u)
        t_in, t_out, t_bath = grid[t["T_in"]], grid[t["T_out"]], grid[t["T_bath"]]
        drive = t_bath - t_in
        eps = (t_out - t_in) / drive.where(drive > p_config.EPS_MIN_DRIVE_C)
        eps = eps.where(eps.between(p_config.EPS_VALID_MIN, p_config.EPS_VALID_MAX))

        grid[f"dT_{u}"] = t_out - t_in
        grid[f"drive_{u}"] = drive
        grid[f"eps_{u}"] = eps
        with np.errstate(invalid="ignore", divide="ignore"):
            grid[f"ntu_{u}"] = -np.log(1.0 - eps)

        lo, hi = config.regime_thresholds(u)
        reg = pd.Series(pd.NA, index=grid.index, dtype="object")
        reg[eps <= lo] = "isolated"
        reg[(eps > lo) & (eps < hi)] = "flow"
        reg[eps >= hi] = "lowflow"
        grid[f"regime_{u}"] = reg

        vc = reg.value_counts()
        n = int(vc.sum())
        logger.info("  [%s] ε임계 차단≤%.3f 저유량≥%.3f | 판정 %d행 → %s",
                    u, lo, hi, n,
                    ", ".join(f"{k} {100*vc.get(k,0)/max(n,1):.1f}%" for k in REGIMES))
    return grid


def run() -> pd.DataFrame:
    grid = build_grid(load_raw())
    grid = add_burner_state(grid)
    grid = add_train_features(grid)
    grid = add_unit_features(grid)
    return grid
