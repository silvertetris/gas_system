"""시간순 8:2 분할 · 개발구간 TimeSeriesSplit · fold 경계별 정제 재적합.

## 왜 정제를 fold 마다 다시 적합하나

AE·EKF 는 **적합하는 모수가 있다**(AE 가중치·정규화 통계, EKF 물리모수·잡음). 기존 산출물은
2019 년까지로 적합돼 있어, 검증창이 2015~2019 인 초기 fold 에 **미래 정보가 샌다.**
그래서 fold 경계마다 **그 fold 학습 구간 끝까지만** 적합한 파일을 따로 만든다.
KF 는 고정 이득(0.05)이라 적합할 모수가 없어 한 번만 계산한다(`risk/refine.build_many`).

최종 모델용 경계(`final`)는 개발구간 **앞 90% 끝**이다 — 뒤 10% 로 조기종료·등장성 보정을 하므로
그 구간이 정제 적합에 들어가면 보정 데이터가 표본 밖이 아니게 된다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import config, data, ekf

log = logging.getLogger(__name__)
PURGE_ROWS = config.rows(config.PURGE_H + config.HORIZON_H)      # 해상도에 맞춘 행 수
PURGE_TD = pd.Timedelta(hours=config.PURGE_H + config.HORIZON_H)


def base_frame(train: str, target: str = "min_raw"):
    """정제 없는 기반 특징. VALID_FROM·핵심 결측·타깃 결측만 거른다(시간 마스크는 쓰지 않는다)."""
    f = data.build(train, refine=(), target=target)
    X, cols, *_ = data.split(f)
    return X, cols


def dev_test(X: pd.DataFrame):
    """시간순 8:2. 경계 앞 퍼지(30h)만큼 개발구간에서 버린다."""
    t_cut = X.index[int(len(X) * (1 - config.TEST_RATIO))]
    dev = np.asarray(X.index < t_cut - PURGE_TD)
    test = np.asarray(X.index >= t_cut)
    return dev, test, t_cut


def cv_folds(X_dev: pd.DataFrame, n_splits: int = config.N_SPLITS) -> list[dict]:
    """TimeSeriesSplit(gap=퍼지). 항상 앞을 학습, 뒤를 검증."""
    n = len(X_dev)
    out = []
    for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits, gap=PURGE_ROWS).split(np.arange(n)), 1):
        trm = np.zeros(n, bool); trm[tr] = True
        vam = np.zeros(n, bool); vam[va] = True
        out.append({"tag": f"fold{k}", "tr": trm, "va": vam,
                    "fit_end": X_dev.index[tr[-1]] + config.STEP_TD})
    return out


def inner_split(X_dev: pd.DataFrame, ratio: float = config.INNER_VAL_RATIO):
    """최종 학습용: 개발구간 앞 90% 학습 / 뒤 10% 조기종료·보정 (퍼지 포함)."""
    t_cut = X_dev.index[int(len(X_dev) * (1 - ratio))]
    tr = np.asarray(X_dev.index < t_cut - PURGE_TD)
    va = np.asarray(X_dev.index >= t_cut)
    return tr, va, X_dev.index[tr][-1] + config.STEP_TD


def kfae_path(train: str, tag: str):
    return config.FOLD_DIR / f"refined_{train}_{tag}.parquet"


def ekf_path(train: str, tag: str):
    return config.FOLD_DIR / f"ekf_{train}_{tag}.parquet"


def ensure_refinements(train: str, fit_ends: dict, force: bool = False) -> None:
    """경계별 KF·AE(한 번에) · EKF 산출. 이미 있으면 건너뛴다."""
    from risk import refine
    config.FOLD_DIR.mkdir(parents=True, exist_ok=True)
    need = {t: fe for t, fe in fit_ends.items() if force or not kfae_path(train, t).exists()}
    if need:
        log.info("[%s] KF·AE 경계별 재적합 %d개: %s", train, len(need),
                 {k: str(v) for k, v in need.items()})
        refine.build_many(train, need, config.FOLD_DIR, freq=config.FREQ)
    for tag, fe in fit_ends.items():
        if force or not ekf_path(train, tag).exists():
            log.info("[%s] EKF 재적합 %s (적합 끝 %s)", train, tag, fe)
            ekf.run(train, fit_end=fe, out_path=ekf_path(train, tag),
                    params_path=config.FOLD_DIR / f"ekf_params_{train}_{tag}.csv")


def load_refinements(train: str, tags: list[str], smoke: bool = False) -> dict:
    """경계 태그 → 정제 13열 프레임. `smoke` 면 기존 기본 산출물을 모든 태그에 쓴다(배선 점검용)."""
    frames = {}
    for tag in tags:
        kp = config.REFINED_KFAE[train] if smoke else kfae_path(train, tag)
        ep = config.REFINED_EKF[train] if smoke else ekf_path(train, tag)
        kfae = pd.read_parquet(kp, columns=config.REFINE_COLS["kf"] + config.REFINE_COLS["ae"])
        frames[tag] = kfae.join(pd.read_parquet(ep, columns=config.REFINE_COLS["ekf"]), how="outer")
    return frames
