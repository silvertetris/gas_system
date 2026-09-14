"""특징·타깃 생성. 1시간 집계, 누수 방지 시간분할."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from prex_multi import config as m_config

from . import config

logger = logging.getLogger(__name__)


def load_raw() -> pd.DataFrame:
    """전처리 parquet + 원본에서 출구온도·압력만 추가로 읽는다."""
    need = {"Time", config.P_IN}
    for t in config.TRAINS.values():
        need |= {t["t61"], t["p61"], t["hdr"], t["t_in"]}
        for u in t["units"]:
            need |= {f"TI-D2{u}", f"TI33{u}"}
    g = pd.read_parquet(config.TREND_PARQUET)
    have = set(g.columns)
    g = g.set_index("Time").sort_index()

    missing = sorted((need - {"Time"}) - have)
    if missing:                       # TI61x·PI61x·PI21X 는 prex_multi 산출물에 없다
        frames = []
        for p in sorted(config.RAW_TREND_DIR.glob("*.csv")):
            d = pd.read_csv(p, usecols=["Time", *missing], low_memory=False)
            d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
            frames.append(d.dropna(subset=["Time"]))
        extra = pd.concat(frames, ignore_index=True)
        extra = extra.drop_duplicates("Time", keep="last").set_index("Time").sort_index()
        extra = extra.apply(pd.to_numeric, errors="coerce")
        g = g.join(extra, how="left")
    return g


def build(train: str) -> pd.DataFrame:
    """계열 하나의 시간 단위 특징·타깃."""
    spec = config.TRAINS[train]
    g = load_raw()
    idx = g.index.floor(config.FREQ)
    t61 = g[spec["t61"]]

    f = pd.DataFrame({
        # --- 현재 상태 (예측 시점까지만 사용 — 누수 없음)
        "t61": t61.groupby(idx).mean(),
        "t61_min": t61.groupby(idx).min(),
        "hdr": g[spec["hdr"]].where(g[spec["hdr"]] > 0).groupby(idx).mean(),
        "t_in": g[spec["t_in"]].groupby(idx).mean(),
        "p_in": g[config.P_IN].groupby(idx).mean(),
        "p61": g[spec["p61"]].groupby(idx).mean(),
    })
    f["dp"] = f["p_in"] - f["p61"]                       # 압력강하 → 줄-톰슨 냉각량
    f["jt_drop"] = f["hdr"] - f["t61"]                   # 실측 강하 (JT + 배관손실)

    for u in spec["units"]:
        f[f"bath_{u}"] = g[f"TI-D2{u}"].groupby(idx).mean()
        f[f"out_{u}"] = g[f"TI33{u}"].groupby(idx).mean()
        f[f"duty_{u}"] = g[f"burner_{u}"].groupby(idx).mean()
        f[f"flow_{u}"] = (g[f"regime_{u}"] == "flow").astype(float).groupby(idx).mean()
    f[f"beta"] = g[f"beta_{train}"].groupby(idx).mean()
    f["valve"] = g[f"valve_{train}"].groupby(idx).mean()
    if config.USE_HEATFLOW:
        f = add_heatflow(f, train)

    # --- 이력 (과거만)
    for w in (3, 12, 24):
        f[f"t61_mean{w}"] = f["t61"].rolling(w, min_periods=max(w // 2, 1)).mean()
        f[f"t61_min{w}"] = f["t61"].rolling(w, min_periods=max(w // 2, 1)).min()
    f["t61_trend6"] = f["t61"] - f["t61"].shift(6)
    f["t_in_mean24"] = f["t_in"].rolling(24, min_periods=12).mean()

    # --- 계절/시각
    f["month"] = f.index.month
    f["hour"] = f.index.hour
    f["doy_sin"] = np.sin(2 * np.pi * f.index.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * f.index.dayofyear / 365.25)

    # --- 타깃: 향후 H시간 내 최저 출구온도 (엄격히 미래)
    for h in config.HORIZONS_H:
        fut = f["t61_min"].shift(-1).rolling(h, min_periods=max(h // 2, 1)).min().shift(-(h - 1))
        f[f"y_min{h}"] = fut
        f[f"y_margin{h}"] = fut - config.FREEZE_C
        f[f"y_event{h}"] = (fut < config.FREEZE_C).astype("float")
    return f


def add_heatflow(f: pd.DataFrame, train: str) -> pd.DataFrame:
    """수조 열수지 유량(문서 12)을 시간 특징으로 붙인다.

    문서 08 §7-6 의 한계 — "상류·하류 계측이 계열 공용이라 **히터 개별 기여가 분리되지
    않는다**" — 를 푸는 열쇠다. 사이클 단위(히터당 시간당 약 1개)라 시간격자에 맞추고
    **과거 방향으로만** 채운다(누수 방지, 최대 6시간).
    """
    spec = config.TRAINS[train]
    path = config.HEATFLOW_PARQUET
    if not path.exists():
        logger.warning("heatflow 산출물이 없다 — 특징을 건너뛴다: %s", path)
        return f
    c = pd.read_parquet(path)
    c = c[c["계열"] == train]
    for u in spec["units"]:
        cu = c[c["히터"] == u].set_index("t0").sort_index()
        if not len(cu):
            continue
        for src, name in (("m_kg_s", "m"), ("fire", "fire"), ("q_gas_kw", "qgas")):
            s = cu[src].groupby(cu.index.floor(config.FREQ)).mean()
            f[f"{name}_{u}"] = s.reindex(f.index).ffill(limit=6)
    ms = [f[f"m_{u}"] for u in spec["units"] if f"m_{u}" in f]
    if ms:
        f["m_tot"] = sum(x.fillna(0.0) for x in ms)
        f["m_tot"] = f["m_tot"].where(pd.concat(ms, axis=1).notna().any(axis=1))
        # 열수지 유량 대비 실제 승온 — 히터가 부하를 따라가는가
        f["heat_margin"] = f["hdr"] - f["t_in"]
    return f


FEATURES_EXCLUDE = {"t61_min"}     # 타깃 생성에 직접 쓰인 열은 특징에서 뺀다


def split(f: pd.DataFrame, h: int):
    """시간 분할 + purge. 경계에서 지평만큼 버려 타깃 누수를 막는다."""
    if config.REGIME_FROM:
        # ⚠ 2015~2016 에 **운전 체제가 바뀌었다**(문서 15) — 바이패스 밸브 0→11~18,
        #   통가스 히터 0.24→1.00대. 구 체제를 학습에서 뺄지 선택한다.
        #   **X 를 만들기 전에** 잘라야 한다 (뒤에서 자르면 마스크 길이가 어긋난다).
        f = f[f.index >= pd.Timestamp(config.REGIME_FROM)]
    cols = [c for c in f.columns if not c.startswith("y_") and c not in FEATURES_EXCLUDE]
    y = f[f"y_margin{h}"]
    # ⚠ **특징 결측으로 행을 버리지 않는다.** HistGradientBoosting 은 NaN 을 원생 처리한다.
    #   열수지 특징(문서 12)은 사이클 단위라 시간 커버리지가 29~62% 뿐이다. 모든 특징이
    #   갖춰진 행만 남기면 시험표본이 32,333 → 4,564 로 붕괴하고 비교가 불가능해진다.
    #   **핵심 특징**(원래부터 있던 계측)만 결측을 걸러 체제 밖 행을 제거한다.
    core = [c for c in cols if not c.startswith(("m_", "fire_", "qgas_", "m_tot", "heat_margin"))]
    ok = y.notna() & f[core].notna().all(axis=1)
    X, y = f.loc[ok, cols], y[ok]
    p = pd.Timedelta(hours=config.PURGE_H + h)
    tr = X.index < pd.Timestamp(config.TRAIN_END) - p
    va = (X.index >= pd.Timestamp(config.TRAIN_END)) & (X.index < pd.Timestamp(config.VALID_END) - p)
    te = X.index >= pd.Timestamp(config.VALID_END)
    return X, y, tr, va, te, cols
