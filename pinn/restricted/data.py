"""전 기간 1시간 특징·타깃. 원본 CSV 는 한 번만 읽고 캐시한다."""
from __future__ import annotations

import functools
import logging
import pathlib

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _raw_hourly() -> pd.DataFrame:
    """`TI61x`·`PI61x`·`PI21X` 는 prex_multi 산출물에 없어 원본에서 읽는다 (175개 파일)."""
    if config.RAW_CACHE.exists():
        return pd.read_parquet(config.RAW_CACHE)
    cols = [config.P_IN] + [s[k] for s in config.TRAINS.values() for k in ("t61", "p61")]
    frames = []
    for p in sorted(config.RAW_TREND_DIR.glob("*.csv")):
        d = pd.read_csv(p, usecols=["Time", *cols], low_memory=False)
        d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
        frames.append(d.dropna(subset=["Time"]))
    g = (pd.concat(frames, ignore_index=True).drop_duplicates("Time", keep="last")
           .set_index("Time").sort_index().apply(pd.to_numeric, errors="coerce"))
    idx = g.index.floor(config.FREQ)
    out = g.groupby(idx).mean()
    for s in config.TRAINS.values():
        out[f"{s['t61']}_min"] = g[s["t61"]].groupby(idx).min()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.RAW_CACHE)
    logger.info("원본 시간집계 캐시 저장: %s (%d행)", config.RAW_CACHE, len(out))
    return out


def _heat_flow(train: str, index: pd.DatetimeIndex, beta: pd.Series) -> tuple[pd.Series, pd.Series]:
    """수조 열수지 사이클 유량 → 계열 총유량 [kg/s] 과 관측 플래그."""
    c = pd.read_parquet(config.HEATFLOW_PARQUET)
    c = c[c["계열"] == train]
    per = []
    for u in config.TRAINS[train]["units"]:
        cu = c[c["히터"] == u]
        s = cu.groupby(cu["t0"].dt.floor(config.FREQ))["m_kg_s"].mean()
        per.append(s.reindex(index).ffill(limit=config.rows(config.M_FFILL_H)).rename(u))
    per = pd.concat(per, axis=1)
    obs = per.notna().any(axis=1)
    m_heat = per.fillna(0.0).sum(axis=1).where(obs)
    # 계열 배관은 히터 통과분 + 바이패스를 모두 싣는다
    m_tot = m_heat / (1.0 - beta.clip(0.0, config.BETA_MAX).fillna(0.0))
    m_tot = m_tot.clip(upper=config.M_CAP_X_DESIGN * config.M_DESIGN_TOT[train])
    return m_tot, obs.astype(float)


def build(train: str, refine: tuple | None = None, kfae_path=None, ekf_path=None,
          target: str = "min_raw") -> pd.DataFrame:
    """기반 특징 + 선택된 정제 신호. 정제 신호는 시각 t 까지의 인과 산출이라 누수가 없다.

    `refine`·`kfae_path`·`ekf_path` 를 주면 config 기본값 대신 쓴다 — Optuna 가 정제 조합을
    고르고, **fold 경계마다 따로 적합한** AE·EKF 산출물을 가리키게 하기 위함.
    """
    return add_refine(build_base(train, target).copy(), train, refine, kfae_path, ekf_path)


def add_refine(f: pd.DataFrame, train: str, refine: tuple | None = None,
               kfae_path=None, ekf_path=None) -> pd.DataFrame:
    """AE·KF(`risk/refine.py`, 1분→1시간) · EKF(`ekf.py`, 1시간) 산출을 붙인다.

    ⚠ 정제 도구다 — 예측기가 아니다. 결측은 그대로 두고 `train.py` 가 0 대치 + 결측 플래그를 단다.
    """
    refine = config.REFINE if refine is None else refine
    for name in refine:
        if name == "ekf":
            src = ekf_path or config.REFINED_EKF[train]
        else:
            src = kfae_path or config.REFINED_KFAE[train]
        if not pathlib.Path(src).exists():
            raise FileNotFoundError(f"정제 산출물이 없다: {src} "
                                    f"({'python -m pinn.restricted.ekf' if name == 'ekf' else 'python -m risk.refine'})")
        r = pd.read_parquet(src, columns=config.REFINE_COLS[name])
        f = f.join(r, how="left")
    return f


@functools.lru_cache(maxsize=None)
def build_base(train: str, target: str = "min_raw") -> pd.DataFrame:
    """무거운 1분→1시간 집계. 정제 조합이 바뀌어도 다시 하지 않는다.

    `target` (문서 19):
      · min_raw — 기존. 6h 창의 **분 단위 최저** T61, 센서 이상값 그대로
      · min     — 같은 타깃, 단 분 단위 최저가 센서 하한(≤ −29.5℃)인 시간은 t61·t61_min 을 결측 처리
      · mean    — 6h 창의 **1시간 평균** 최저 T61 (물리식이 기술하는 해상도), 이상값 제거 동일
    ⚠ 하류 코드를 바꾸지 않으려고 타깃 값은 어느 경우든 `y_t61_min` 열에 둔다(mean 이면 평균 최저).
    `y_t61_mean` = 타깃 시각의 1시간 평균 T61 (soft PINN 의 순간하강 d 지도용).
    """
    assert target in config.TARGETS, target
    spec = config.TRAINS[train]
    keep = ["Time", spec["hdr"], spec["t_in"], f"beta_{train}", f"valve_{train}"]
    for u in spec["units"]:
        keep += [f"TI-D2{u}", f"TI33{u}", f"burner_{u}"]
    g = pd.read_parquet(config.TREND_PARQUET, columns=keep).set_index("Time").sort_index()
    idx = g.index.floor(config.FREQ)
    raw = _raw_hourly()

    f = pd.DataFrame({
        "hdr": g[spec["hdr"]].where(g[spec["hdr"]] > 0).groupby(idx).mean(),
        "t_in": g[spec["t_in"]].groupby(idx).mean(),
        "beta": g[f"beta_{train}"].groupby(idx).mean(),
        "valve": g[f"valve_{train}"].groupby(idx).mean(),
    })
    for u in spec["units"]:
        f[f"bath_{u}"] = g[f"TI-D2{u}"].groupby(idx).mean()
        f[f"out_{u}"] = g[f"TI33{u}"].groupby(idx).mean()
        f[f"duty_{u}"] = g[f"burner_{u}"].groupby(idx).mean()
    f = f.join(raw[[spec["t61"], f"{spec['t61']}_min", spec["p61"], config.P_IN]], how="left")
    f = f.rename(columns={spec["t61"]: "t61", f"{spec['t61']}_min": "t61_min",
                          spec["p61"]: "p61", config.P_IN: "p_in"})
    if target != "min_raw":
        # 1분 급락이 측정 하한(−30℃)에 찍힌 시간: 그 시간 평균도 오염되므로 둘 다 결측 (M 39·Z 32시간)
        bad = f["t61_min"] <= config.T61_SENSOR_FLOOR
        f.loc[bad, ["t61", "t61_min"]] = np.nan
    f["dp"] = (f["p_in"] - f["p61"]) * config.MPA_TO_BAR          # bar
    f["m_tot"], f["m_obs"] = _heat_flow(train, f.index, f["beta"])
    f["log_m"] = np.log(f["m_tot"].clip(lower=config.M_MIN))

    # --- 이력·계절 (과거만)
    # 이름의 숫자는 **시간**이다. 창 길이는 해상도에 맞춰 행 수로 바꾼다(60분이면 이전과 같다).
    for w in (3, 12, 24):
        n = config.rows(w)
        f[f"t61_mean{w}"] = f["t61"].rolling(n, min_periods=max(n // 2, 1)).mean()
        f[f"t61_minr{w}"] = f["t61_min"].rolling(n, min_periods=max(n // 2, 1)).min()
        f[f"hdr_mean{w}"] = f["hdr"].rolling(n, min_periods=max(n // 2, 1)).mean()
    f["t61_trend6"] = f["t61"] - f["t61"].shift(config.rows(6))
    f["t_in_mean24"] = f["t_in"].rolling(config.rows(24), min_periods=config.rows(12)).mean()
    f["doy_sin"] = np.sin(2 * np.pi * f.index.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * f.index.dayofyear / 365.25)
    f["hour_sin"] = np.sin(2 * np.pi * f.index.hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * f.index.hour / 24.0)

    # --- 타깃: 향후 H시간 **최저 공급온도가 난 그 시각**의 상태로 전부 정렬
    #   (상태마다 따로 극값을 잡으면 서로 다른 시점을 한 식에 넣게 된다 — 문서 09 §8-3)
    h = config.rows(config.HORIZON_H)          # 예측 창(행). 60분이면 6

    def windows(a: np.ndarray) -> np.ndarray:
        x = np.concatenate([a[1:].astype(np.float64), np.full(h, np.nan)])
        return np.lib.stride_tricks.sliding_window_view(x, h)[:len(a)]

    tw = windows(f["t61" if target == "mean" else "t61_min"].to_numpy())
    valid = np.isfinite(tw).sum(1) >= max(h // 2, 1)
    arg = np.where(valid, np.nanargmin(np.where(np.isfinite(tw), tw, np.inf), axis=1), 0)
    rows = np.arange(len(f))

    def at_min(col: str) -> np.ndarray:
        return np.where(valid, windows(f[col].to_numpy())[rows, arg], np.nan)

    f["y_t61_min"] = np.where(valid, tw[rows, arg], np.nan)
    f["y_t61_mean"] = at_min("t61")
    f["y_hdr"], f["y_dp"], f["y_tin"] = at_min("hdr"), at_min("dp"), at_min("t_in")
    f["y_log_m"] = np.where(at_min("m_obs") > 0.5, at_min("log_m"), np.nan)
    f["y_event"] = (f["y_t61_min"] < config.FREEZE_C).astype(float)
    return f


# 신경망 입력. 타깃·파생 플래그 외 전부. `t61_min` 은 타깃 생성에 쓰였으나 **현재 시각**
# 값이므로 누수가 아니다(y_ 는 미래 창).
EXCLUDE = {"m_tot"}            # log_m 과 중복


def split(f: pd.DataFrame):
    f = f[f.index >= pd.Timestamp(config.VALID_FROM)]
    if config.REGIME_FROM:
        f = f[f.index >= pd.Timestamp(config.REGIME_FROM)]
    cols = [c for c in f.columns if not c.startswith("y_") and c not in EXCLUDE]
    core = ["t61", "t61_min", "hdr", "t_in", "dp"]
    need = ["y_t61_min", "y_hdr", "y_dp", "y_tin"]
    ok = f[core + need].notna().all(axis=1)
    X = f.loc[ok]
    p = pd.Timedelta(hours=config.PURGE_H + config.HORIZON_H)
    tr = np.asarray(X.index < pd.Timestamp(config.TRAIN_END) - p)
    va = np.asarray((X.index >= pd.Timestamp(config.TRAIN_END))
                    & (X.index < pd.Timestamp(config.VALID_END) - p))
    te = np.asarray(X.index >= pd.Timestamp(config.VALID_END))
    return X, cols, tr, va, te
