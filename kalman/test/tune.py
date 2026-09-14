"""Optuna 탐색 — TimeSeriesSplit 5-fold, 1스텝 예측 RMSE 최소화.

튜닝 대상은 KF 의 고전 하이퍼파라미터인 **Q(프로세스 잡음)와 R(관측 잡음)** 이다.
학습(=Q/R 적합)에 테스트 구간이 들어가지 않도록 fold 를 시간순으로 자른다.
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

from . import config, data, model

logger = logging.getLogger(__name__)


def _suggest(trial: optuna.Trial) -> dict:
    sp = config.SEARCH_SPACE
    p = {
        "model_type": trial.suggest_categorical("model_type", sp["model_type"]),
        "q_level": trial.suggest_float("q_level", *sp["q_level"], log=True),
        "r_scale": trial.suggest_float("r_scale", *sp["r_scale"], log=True),
    }
    # 기울기 잡음은 local_trend 에서만 의미가 있다
    p["q_slope"] = (trial.suggest_float("q_slope", *sp["q_slope"], log=True)
                    if p["model_type"] == "local_trend" else 0.0)
    return p


def cv_score(params: dict, Z_raw: np.ndarray, n_splits: int = config.N_SPLITS) -> float:
    """fold 별 1스텝 예측 RMSE(표준화 단위) 평균.

    ⚠ fold 마다 스케일러를 그 fold 의 학습부분으로만 fit 한다.
    ⚠ KF 는 인과 필터라 검증 구간을 **이어서 돌려야** 초기 과도상태가 생기지 않는다.
       그래서 학습+검증을 연속으로 필터링하되, **점수는 검증 구간에서만** 매긴다.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=config.PURGE_MIN)
    scores = []
    for tr_idx, va_idx in tscv.split(Z_raw):
        xs = data.Scaler().fit(Z_raw[tr_idx][None, ...])       # (1, T, S) 형태로 맞춰 fit
        run_idx = np.arange(tr_idx[0], va_idx[-1] + 1)          # 연속 구간
        Zs = xs.transform(Z_raw[run_idx][None, ...])[0]
        res = model.run_filter(Zs, params, dt=config.DT)
        local_va = va_idx - run_idx[0]
        rmse, _ = model.one_step_rmse(Zs[local_va], res["predicted"][local_va])
        scores.append(rmse)
    return float(np.mean(scores))


def run_search(Z_raw: np.ndarray, n_trials: int = config.N_TRIALS) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        study_name="kf_pilot",
    )

    def objective(trial: optuna.Trial) -> float:
        p = _suggest(trial)
        s = cv_score(p, Z_raw)
        logger.info("trial %2d  CV 1-step RMSE %.5f  %s", trial.number, s, p)
        return s

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("최적 CV %.5f | %s", study.best_value, study.best_params)
    return study
