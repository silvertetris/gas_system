"""검정용 데이터 — 1분 원해상도, 전 기간. 히터 블록만 쓴다.

⚠ **시간평균을 쓰지 않는다.** 1시간 집계는 합류점식을 깨뜨린다 — `T_hdr` 이
`{T_out,u, T_in}` 의 볼록껍질 밖으로 나가는 행이 M 16.9% · Z 31.0% 였다(문서 09 §5-1).
가중합의 Jensen 오차와 분(分) 단위 regime 혼재가 원인이므로 **원해상도로 검정한다.**

배관 구간(`Q5` 줄-톰슨 · `Q6` 지중 열교환)은 여기서 다루지 않는다 — 이미 잔차구조가
붕괴하는 것으로 확인됐고(문서 09), 원본 CSV 를 다시 읽어야 해서 비용이 다르다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def build(train: str) -> pd.DataFrame:
    spec = config.TRAINS[train]
    units = spec["units"]

    keep = ["Time", spec["t_in"], spec["hdr"], spec["valve"]]
    for u in units:
        keep += [f"TI33{u}", f"TI-D2{u}", f"burner_{u}", f"regime_{u}"]
    g = pd.read_parquet(config.TREND_PARQUET, columns=keep).set_index("Time").sort_index()
    g = g.loc[pd.Timestamp(config.VALID_FROM):]

    d = pd.DataFrame(index=g.index)
    d["t_in"] = g[spec["t_in"]]
    d["hdr"] = g[spec["hdr"]].where(g[spec["hdr"]] > 0)
    d["valve"] = g[spec["valve"]]
    for u in units:
        d[f"out_{u}"] = g[f"TI33{u}"]
        d[f"bath_{u}"] = g[f"TI-D2{u}"]
        d[f"duty_{u}"] = g[f"burner_{u}"]
        d[f"regime_{u}"] = g[f"regime_{u}"]
        # ε 은 세 온도로 직접 계산된다 — 추정이 아니다
        drive = d[f"bath_{u}"] - d["t_in"]
        eps = (d[f"out_{u}"] - d["t_in"]) / drive.where(drive > config.MIN_DRIVE_C)
        d[f"eps_{u}"] = eps.where(eps.between(config.EPS_MIN, config.EPS_MAX))
        d[f"drive_{u}"] = drive

    need = ["t_in", "hdr", "valve"] + [f"{p}_{u}" for u in units
                                       for p in ("out", "bath", "eps", "duty")]
    d = d.dropna(subset=need)

    # 계절·시각 (유량의 외생 설명변수. `T_out`·`T_hdr` 은 절대 특징으로 쓰지 않는다)
    d["doy_sin"] = np.sin(2 * np.pi * d.index.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.index.dayofyear / 365.25)
    d["hour_sin"] = np.sin(2 * np.pi * d.index.hour / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * d.index.hour / 24.0)

    logger.info("[%s] %s ~ %s, %d행 (ε 유효)", train,
                d.index[0].date(), d.index[-1].date(), len(d))
    return d


# 유량 잠재를 내는 신경망의 입력. **`out_*`·`hdr` 은 제외** — 그게 맞혀야 하는 대상이다.
def feature_cols(train: str) -> list[str]:
    units = config.TRAINS[train]["units"]
    return (["t_in", "valve", "doy_sin", "doy_cos", "hour_sin", "hour_cos"]
            + [f"bath_{u}" for u in units] + [f"duty_{u}" for u in units])


def split(d: pd.DataFrame):
    tr = d.index < pd.Timestamp(config.TRAIN_END)
    te = d.index >= pd.Timestamp(config.TEST_FROM)
    return np.asarray(tr), np.asarray(te)
