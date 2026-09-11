"""표본 추출 → 윈도우 생성 → 시간순 분할 → 스케일링.

**데이터 누수 방지가 이 모듈의 핵심 책임이다.** 세 군데서 샌다:

  1. 스케일러를 전체 데이터로 fit  → 학습 구간에서만 fit 하고 테스트는 transform 만.
  2. 윈도우가 분할 경계를 걸침     → 경계 앞뒤로 PURGE(=WINDOW+HORIZON)만큼 버린다.
                                      그래야 학습 윈도우의 타깃 시각이 테스트 입력에 안 들어간다.
  3. segment 경계를 넘는 윈도우    → 24시간 이상 결측 구간을 가로지르면 없는 시간을 연결하게 된다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def load_sample() -> pd.DataFrame:
    """표본 구간만 읽고 burner_on 피처를 붙인다."""
    df = pd.read_parquet(config.TREND_PARQUET).set_index("Time").sort_index()
    df = df.loc[config.SAMPLE_START:config.SAMPLE_END]

    ev = pd.read_parquet(config.ALARM_PARQUET)
    h = ev.loc[ev["tag"] == "H31POH", ["Time", "value"]].dropna()
    h = h.sort_values("Time").drop_duplicates("Time", keep="last")
    merged = pd.merge_asof(pd.DataFrame({"Time": df.index}), h, on="Time", direction="backward")
    # ⚠ H31POH 는 "히터 가동 기간"이 아니라 버너 점화 사이클이다(ON 중앙 15분).
    #    docs/htr31p_flow.md §8-4 참고. 여기서는 구동항 게이트로만 쓴다.
    df["burner_on"] = (merged["value"].to_numpy() == 1.0).astype("float64")

    keep = list(dict.fromkeys(config.FEATURES + [config.TARGET, "segment_id"]))
    df = df[keep]
    logger.info("표본 로드: %s ~ %s, %d행, 결측 %.2f%%",
                df.index.min(), df.index.max(), len(df),
                100 * df[config.FEATURES].isna().any(axis=1).mean())
    return df


def make_windows(df: pd.DataFrame, window: int | None = None, horizon: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """(N, L, F) 입력과 (N,) 타깃을 만든다.

    윈도우 i 는 [t_i-L+1, t_i] 를 입력으로 t_i+H 의 TARGET 을 예측한다.
    - 입력·타깃 구간에 NaN 이 하나라도 있으면 그 윈도우는 버린다(보간 금지 원칙).
    - 입력~타깃이 같은 segment 안에 있어야 한다.

    Returns: X, y, 윈도우 시작시각, 타깃시각
    """
    L = config.WINDOW if window is None else window
    H = config.HORIZON if horizon is None else horizon
    feats = df[config.FEATURES].to_numpy(dtype="float32")
    target = df[config.TARGET].to_numpy(dtype="float32")
    seg = df["segment_id"].to_numpy()
    n = len(df)

    # 슬라이딩 윈도우 뷰 (복사 없이) — 마지막 인덱스가 t_i 인 윈도우들
    ends = np.arange(L - 1, n - H)
    valid_feat = ~np.isnan(feats).any(axis=1)
    # 윈도우 전체가 유효해야 하므로 누적합으로 구간 유효성 판정
    cum = np.concatenate([[0], np.cumsum(valid_feat)])
    win_ok = (cum[ends + 1] - cum[ends + 1 - L]) == L
    tgt_ok = ~np.isnan(target[ends + H])
    seg_ok = seg[ends + 1 - L] == seg[ends + H]        # 시작~타깃이 같은 segment
    ok = win_ok & tgt_ok & seg_ok
    ends = ends[ok]

    idx = ends[:, None] + np.arange(-L + 1, 1)[None, :]
    X = feats[idx]                                      # (N, L, F)
    y = target[ends + H]                                # (N,)
    logger.info("윈도우 %d개 생성 (후보 %d개 중 %.1f%% 유효), X%s",
                len(ends), ok.size, 100 * len(ends) / max(ok.size, 1), X.shape)
    return X, y, df.index[ends + 1 - L], df.index[ends + H]


def split_purged(n: int, test_ratio: float = config.TEST_RATIO,
                 purge: int = config.PURGE) -> tuple[np.ndarray, np.ndarray]:
    """시간순 8:2 분할 + 경계 purge.

    윈도우는 시간순으로 정렬돼 있으므로 인덱스를 자르면 시간 분할이 된다.
    경계 앞뒤 purge 개를 **버려서** 학습 윈도우의 타깃 시각(t+H)이
    테스트 윈도우의 입력 구간(t-L+1..t)과 절대 겹치지 않게 한다.
    """
    cut = int(n * (1 - test_ratio))
    train = np.arange(0, max(cut - purge, 0))
    test = np.arange(min(cut + purge, n), n)
    logger.info("분할: train %d / purge %d(버림) / test %d",
                len(train), n - len(train) - len(test), len(test))
    return train, test


class Scaler:
    """학습 구간 통계로만 fit 하는 robust 스케일러 (중앙값/IQR).

    왜 robust 인가: 이 데이터는 왜도가 크다(RSF41P skew −8.8, PI21X −4.5 — prex/profiling.py).
    평균/표준편차를 쓰면 소수의 계기 dropout 이 스케일을 끌고 간다.
    """

    def __init__(self) -> None:
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Scaler":
        flat = X.reshape(-1, X.shape[-1])
        q25, q50, q75 = np.percentile(flat, [25, 50, 75], axis=0)
        iqr = q75 - q25
        self.center_ = q50
        self.scale_ = np.where(iqr > 1e-9, iqr, 1.0)     # 상수열(예: burner_on) 보호
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.center_) / self.scale_).astype("float32")


class TargetScaler(Scaler):
    """타깃(1차원)용. 역변환이 필요해 별도로 둔다."""

    def fit(self, y: np.ndarray) -> "TargetScaler":
        return super().fit(y.reshape(-1, 1))

    def transform(self, y: np.ndarray) -> np.ndarray:
        return super().transform(y.reshape(-1, 1)).ravel()

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.scale_[0] + self.center_[0]
