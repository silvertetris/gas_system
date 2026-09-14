"""Optuna 탐색 — TimeSeriesSplit 5-fold. lstm.test.tune 과 같은 규칙."""
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
        "latent_dim": trial.suggest_categorical("latent_dim", sp["latent_dim"]),
        "hidden_size": trial.suggest_categorical("hidden_size", sp["hidden_size"]),
        "num_layers": trial.suggest_categorical("num_layers", sp["num_layers"]),
        "dropout": trial.suggest_float("dropout", *sp["dropout"]),
        "lr": trial.suggest_float("lr", *sp["lr"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", sp["batch_size"]),
        "weight_decay": trial.suggest_float("weight_decay", *sp["weight_decay"], log=True),
    }


def cv_score(params: dict, X: np.ndarray, n_splits: int = config.N_SPLITS,
             epochs: int = config.TUNE_EPOCHS) -> float:
    """fold 별 재구성 RMSE(원 단위) 평균.

    ⚠ fold 마다 스케일러를 그 fold 의 학습부분으로만 새로 fit 한다(누수 방지).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=config.PURGE)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        xs = data.Scaler().fit(X[tr_idx])
        m, _ = model.train_model(xs.transform(X[tr_idx]), xs.transform(X[va_idx]),
                                 params, epochs=epochs)
        Xs_va = xs.transform(X[va_idx])
        rec = model.reconstruct(m, Xs_va)
        # 표준화 공간에서 채점한다 — 학습이 최적화하는 공간과 같아야 탐색이 일관된다.
        # (원 단위로 재면 분산 큰 신호가 지표를 지배해 다른 걸 튜닝하게 된다)
        scores.append(float(np.sqrt(np.mean((rec - Xs_va) ** 2))))
    return float(np.mean(scores))


def run_search(X: np.ndarray, n_trials: int = config.N_TRIALS) -> optuna.Study:
    model.set_seed(config.SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        study_name="ae_pilot",
    )

    def objective(trial: optuna.Trial) -> float:
        p = _suggest(trial)
        s = cv_score(p, X)
        logger.info("trial %2d  CV recon RMSE %.4f  %s", trial.number, s, p)
        return s

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("최적 CV %.4f | %s", study.best_value, study.best_params)
    return study
