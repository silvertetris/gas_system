"""PINN 파일럿 데이터 — 표본 구간만, 1시간 집계, 누수 방지."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

EPS_MIN_DRIVE_C = 5.0    # 구동온도차 하한. 분모가 작으면 ε 이 잡음으로 발산한다

logger = logging.getLogger(__name__)


def _load_extra(cols: list[str], lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """`TI61x`·`PI61x`·`PI21X` 는 prex_multi 산출물에 없다 → 원본에서 표본 구간만 읽는다."""
    frames = []
    for p in sorted(config.RAW_TREND_DIR.glob("*.csv")):
        d = pd.read_csv(p, usecols=["Time", *cols], low_memory=False)
        d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
        d = d.dropna(subset=["Time"])
        d = d[(d["Time"] >= lo) & (d["Time"] < hi)]
        if len(d):
            frames.append(d)
    if not frames:
        raise RuntimeError("표본 구간에 원본 데이터가 없다")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("Time", keep="last")
    return df.set_index("Time").sort_index().apply(pd.to_numeric, errors="coerce")


def build(train: str) -> pd.DataFrame:
    spec = config.TRAINS[train]
    lo, hi = pd.Timestamp(config.SAMPLE_START), pd.Timestamp(config.SAMPLE_END)

    keep = ["Time", spec["hdr"], spec["t_in"], f"beta_{train}", f"valve_{train}"]
    for u in spec["units"]:
        keep += [f"TI-D2{u}", f"TI33{u}", f"burner_{u}", f"regime_{u}"]
    g = pd.read_parquet(config.TREND_PARQUET, columns=keep).set_index("Time").sort_index()
    g = g.loc[lo:hi]
    g = g.join(_load_extra([spec["t61"], spec["p61"], config.P_IN], lo, hi), how="left")

    idx = g.index.floor(config.FREQ)
    t61, hdr = g[spec["t61"]], g[spec["hdr"]].where(g[spec["hdr"]] > 0)
    # ⚠ **단위**: `PI21X`·`PI61x` 는 MPa 다. 줄-톰슨 계수는 ℃/bar 로 쓰므로 **bar 로 변환**한다.
    #   1차 파일럿은 MPa 를 그대로 넣고 μ_JT 를 [0.30,0.50] ℃/bar 로 묶어, 필요한 값의 1/10 에
    #   갇혔다. 물리 잔차가 만족될 수 없어 손실이 210~450 에서 정체하고 예측 헤드를 굶겼다.
    dp = (g[config.P_IN] - g[spec["p61"]]) * config.MPA_TO_BAR

    f = pd.DataFrame({
        # 현재 관측 (예측 시점까지만)
        "t61": t61.groupby(idx).mean(),
        "hdr": hdr.groupby(idx).mean(),
        "t_in": g[spec["t_in"]].groupby(idx).mean(),
        "dp": dp.groupby(idx).mean(),
        "dp_max": dp.groupby(idx).max(),
        "p_in": g[config.P_IN].groupby(idx).mean(),
        "beta": g[f"beta_{train}"].groupby(idx).mean(),
        "valve": g[f"valve_{train}"].groupby(idx).mean(),
    })
    for u in spec["units"]:
        f[f"bath_{u}"] = g[f"TI-D2{u}"].groupby(idx).mean()
        f[f"out_{u}"] = g[f"TI33{u}"].groupby(idx).mean()
        f[f"duty_{u}"] = g[f"burner_{u}"].groupby(idx).mean()
        f[f"flow_{u}"] = (g[f"regime_{u}"] == "flow").astype(float).groupby(idx).mean()
        # (Q1) 저장항 — 수조 승온율 [℃/h]. **과거 차분**만 쓴다(누수 없음).
        f[f"dbath_{u}"] = f[f"bath_{u}"].diff()
        # ε 은 세 온도로 **직접 계산**된다 — 추정이 아니라 관측이다.
        # 분 단위로 먼저 계산하고 시간평균한다 (시간평균 온도로 계산하면 Jensen 오차가 든다).
        drive_m = g[f"TI-D2{u}"] - g[spec["t_in"]]
        eps_m = (g[f"TI33{u}"] - g[spec["t_in"]]) / drive_m.where(drive_m > EPS_MIN_DRIVE_C)
        e = eps_m.where(eps_m.between(config.EPS_LO, config.EPS_HI)).groupby(idx).mean()
        # ⚠ 결측을 남기면 행이 통째로 탈락한다(43,849시간 → 1,376행, 8차 버그). ε 이 정의되지
        #   않는다는 정보는 **구동온도차**에 이미 있고 물리층이 `avail` 로 부드럽게 끈다.
        f[f"eps_{u}"] = e.fillna(0.0)
        f[f"drive_{u}"] = (g[f"TI-D2{u}"] - g[spec["t_in"]]).groupby(idx).mean()

    # 이력·계절
    for w in (3, 12, 24):
        f[f"hdr_mean{w}"] = f["hdr"].rolling(w, min_periods=max(w // 2, 1)).mean()
        f[f"t61_min{w}"] = f["t61"].rolling(w, min_periods=max(w // 2, 1)).min()
    f["hdr_trend6"] = f["hdr"] - f["hdr"].shift(6)
    f["t_in_mean24"] = f["t_in"].rolling(24, min_periods=12).mean()
    f["doy_sin"] = np.sin(2 * np.pi * f.index.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * f.index.dayofyear / 365.25)
    f["hour"] = f.index.hour

    # --- 타깃: 향후 H시간의 **최저 공급온도**와, **그 순간의** 모든 상태
    #
    # ⚠ 이전 판은 상태마다 독립적으로 극값을 잡았다(헤더 최저·압력강하 최대·밸브 최대·
    #   ε 최저). 그러면 **서로 다른 시점의 값들**을 한 물리식에 넣게 되고, 6시간 창의
    #   밸브 최대값처럼 예측 불가능한 타깃이 생긴다(밸브 RMSE 19.9/20.3, ε RMSE 0.19~0.26).
    #   최저 공급온도가 **발생한 그 시각**으로 전부 정렬한다 — 물리적으로도 그게 맞다.
    h = config.HORIZON_H
    def _windows(a: np.ndarray) -> np.ndarray:
        """(n, h) 미래창. t+1 … t+h. 뒤쪽 h칸은 NaN."""
        pad = np.full(h, np.nan, dtype=np.float64)
        x = np.concatenate([a[1:].astype(np.float64), pad])
        return np.lib.stride_tricks.sliding_window_view(x, h)[:len(a)]

    t61w = _windows(f["t61"].to_numpy())
    valid = np.isfinite(t61w).sum(1) >= max(h // 2, 1)
    arg = np.where(valid, np.nanargmin(np.where(np.isfinite(t61w), t61w, np.inf), axis=1), 0)
    rows = np.arange(len(f))

    def at_min(col: str) -> np.ndarray:
        """최저 공급온도 시점의 값."""
        w = _windows(f[col].to_numpy())
        return np.where(valid, w[rows, arg], np.nan)

    f["y_t61_min"] = np.where(valid, t61w[rows, arg], np.nan)
    f["y_hdr_min"] = at_min("hdr")
    f["y_dp_max"] = at_min("dp")
    f["y_tin"] = at_min("t_in")
    f["y_valve"] = at_min("valve")
    for u in spec["units"]:
        f[f"y_bath_{u}"] = at_min(f"bath_{u}")
        f[f"y_eps_{u}"] = at_min(f"eps_{u}")
        f[f"y_drive_{u}"] = at_min(f"drive_{u}")
    f["y_event"] = (f["y_t61_min"] < config.FREEZE_C).astype(float)
    return f


FEATURES_EXCLUDE = {"dp_max"}     # 타깃 생성에 직접 쓰인 열
                                  # `hdr`(현재 실측 헤더온도)는 정당한 입력이다 — 물리층이
                                  # 계산하는 것은 **미래** 헤더온도다.


def split(f: pd.DataFrame):
    cols = [c for c in f.columns if not c.startswith("y_") and c not in FEATURES_EXCLUDE]
    need = cols + ["y_t61_min", "y_hdr_min", "y_dp_max", "y_tin", "y_valve"] + \
        [c for c in f.columns if c.startswith(("y_bath_", "y_eps_", "y_drive_"))]
    ok = f[need].notna().all(axis=1)
    X = f.loc[ok]
    p = pd.Timedelta(hours=config.PURGE_H + config.HORIZON_H)
    tr = X.index < pd.Timestamp(config.TRAIN_END) - p
    va = (X.index >= pd.Timestamp(config.TRAIN_END)) & (X.index < pd.Timestamp(config.VALID_END) - p)
    te = X.index >= pd.Timestamp(config.VALID_END)
    return X, cols, np.asarray(tr), np.asarray(va), np.asarray(te)
