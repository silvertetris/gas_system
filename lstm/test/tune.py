"""Optuna 하이퍼파라미터 탐색 — TimeSeriesSplit 5-fold.

시계열이라 KFold(무작위 셔플)를 쓰면 **미래로 학습해 과거를 맞히는** 누수가 생긴다.
`TimeSeriesSplit` 은 항상 앞을 학습, 뒤를 검증으로 쓴다. 여기에 `gap=PURGE` 를 줘서
학습 윈도우의 타깃 시각이 검증 윈도우 입력에 들어가지 않게 한 번 더 막는다.

⚠ fold 마다 스케일러를 **그 fold 의 학습 부분으로만** 새로 fit 한다.
   전체로 한 번 fit 해 두고 돌리면 검증 구간 통계가 새어 들어간다.
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
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", sp["hidden_size"]),
        "num_layers": trial.suggest_categorical("num_layers", sp["num_layers"]),
        "dropout": trial.suggest_float("dropout", *sp["dropout"]),
        "lr": trial.suggest_float("lr", *sp["lr"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", sp["batch_size"]),
        "weight_decay": trial.suggest_float("weight_decay", *sp["weight_decay"], log=True),
    }


def cv_score(params: dict, X: np.ndarray, y: np.ndarray, n_splits: int = config.N_SPLITS,
             epochs: int = config.TUNE_EPOCHS) -> float:
    """fold 별 검증 RMSE(원 단위 ℃)의 평균."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=config.PURGE)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        xs, ys = data.Scaler().fit(X[tr_idx]), data.TargetScaler().fit(y[tr_idx])
        m, _ = model.train_model(
            xs.transform(X[tr_idx]), ys.transform(y[tr_idx]),
            xs.transform(X[va_idx]), ys.transform(y[va_idx]),
            params, epochs=epochs,
        )
        pred = ys.inverse(model.predict(m, xs.transform(X[va_idx])))
        scores.append(float(np.sqrt(np.mean((pred - y[va_idx]) ** 2))))
    return float(np.mean(scores))


def run_search(X: np.ndarray, y: np.ndarray, n_trials: int = config.N_TRIALS) -> optuna.Study:
    """학습 구간(X, y)만 받아 탐색한다. 테스트 구간은 이 함수에 들어오면 안 된다."""
    model.set_seed()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        study_name="lstm_pilot",
    )

    def objective(trial: optuna.Trial) -> float:
        p = _suggest(trial)
        s = cv_score(p, X, y)
        logger.info("trial %2d  CV RMSE %.4f ℃  %s", trial.number, s, p)
        return s

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("최적 CV RMSE %.4f ℃ | params %s", study.best_value, study.best_params)
    return study
