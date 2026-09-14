"""Optuna — 개발구간 TimeSeriesSplit 5-fold. `lstm/test/tune.py` 와 같은 틀.

  · 목적 = fold 검증 **AUC 평균** (fold 사건률이 0.2~2.5% 로 크게 달라 사건률에 좌우되는 PR-AUC 대신)
  · 정제 조합(8가지)도 하이퍼파라미터로 고른다 — 문서 18 은 시험구간으로 고른 것이라 쓰지 않는다
  · fold 마다 스케일러·결측 대치 통계는 **그 fold 학습부로만**(train.prepare), 정제 산출은
    **그 fold 경계로 재적합한 파일**(folds.load_refinements)
  · MedianPruner — 앞 fold 에서 가망 없는 trial 을 끊는다
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import torch
from sklearn.metrics import roc_auc_score

from . import config, train as T

log = logging.getLogger(__name__)

REFINE_CHOICES = {"없음": (), "KF": ("kf",), "AE": ("ae",), "EKF": ("ekf",),
                  "KF+AE": ("kf", "ae"), "KF+EKF": ("kf", "ekf"), "AE+EKF": ("ae", "ekf"),
                  "KF+AE+EKF": ("kf", "ae", "ekf")}
SPACE = {"hidden": [32, 64, 128], "n_layers": [2, 3], "dropout": (0.0, 0.3),
         "lr": (3e-4, 3e-3), "batch": [1024, 2048, 4096], "weight_decay": (1e-6, 1e-3),
         "w_state": (0.5, 4.0), "w_m": (0.3, 3.0), "w_phys": (0.3, 3.0), "w_cons": (0.3, 3.0),
         "window": [12, 24, 48]}
LSTM_LAYERS = [1, 2]
# softw (2026-09-14): soft 의 w_cons 가 M 10작업 전부에서 CV AUC 와 양의 순위상관(+0.30~+0.64)이고 최적값이
# 상한 3 에 붙었다 → 물리를 더 세게 걸고 싶었는데 범위에 막혔을 수 있다. 범위를 넓혀 soft↔hard 사이를 탐색한다.
SPACE_W = {"w_phys": (0.03, 30.0), "w_cons": (0.3, 30.0)}


VARIANTS = ["없음", "KF", "AE", "EKF", "KF+AE+EKF"]     # 조합별 비교 실행 (넣기 전 / 각각 / 모두)


def suggest(trial: optuna.Trial, fixed: str | None = None, arch: str = "hard") -> tuple[tuple, dict]:
    """`fixed` 가 없으면 정제 조합도 고른다(auto). 있으면 그 조합으로 고정하고 하이퍼파라미터만 찾는다."""
    key = fixed if fixed is not None else trial.suggest_categorical("refine", list(REFINE_CHOICES))
    hp = {"hidden": trial.suggest_categorical("hidden", SPACE["hidden"]),
          "n_layers": trial.suggest_categorical("n_layers", LSTM_LAYERS if arch == "lstm" else SPACE["n_layers"]),
          "dropout": trial.suggest_float("dropout", *SPACE["dropout"]),
          "lr": trial.suggest_float("lr", *SPACE["lr"], log=True),
          "batch": trial.suggest_categorical("batch", SPACE["batch"]),
          "weight_decay": trial.suggest_float("weight_decay", *SPACE["weight_decay"], log=True),
          "w_state": trial.suggest_float("w_state", *SPACE["w_state"], log=True),
          "w_m": trial.suggest_float("w_m", *SPACE["w_m"], log=True)}
    # 제안 순서는 기존(hard)과 같게 유지한다 — w_phys 가 마지막. nn 은 물리 가중치가 없다.
    if arch == "lstm":                           # 과거 창 길이 [시간]
        hp["window"] = trial.suggest_categorical("window", SPACE["window"])
    wr = SPACE_W if arch == "softw" else SPACE
    if arch not in ("nn", "lstm"):
        hp["w_phys"] = trial.suggest_float("w_phys", *wr["w_phys"], log=True)
    if arch in ("soft", "softw"):
        hp["w_cons"] = trial.suggest_float("w_cons", *wr["w_cons"], log=True)
    return REFINE_CHOICES[key], hp


def params_to_hp(best: dict, fixed: str | None = None) -> tuple[tuple, dict]:
    hp = {k: best[k] for k in SPACE if k in best}
    for k in ("hidden", "n_layers", "batch", "window"):
        if k in hp:
            hp[k] = int(hp[k])
    return REFINE_CHOICES[fixed if fixed is not None else best["refine"]], hp


def columns(base_cols: list[str], refine: tuple) -> list[str]:
    return list(base_cols) + [c for r in refine for c in config.REFINE_COLS[r]]


def frame(X, ref, refine: tuple):
    if not refine:
        return X
    return X.join(ref[[c for r in refine for c in config.REFINE_COLS[r]]], how="left")


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def cv_score(train: str, X_dev, base_cols, refs: dict, folds: list[dict], refine: tuple, hp: dict,
             dev: str, trial: optuna.Trial | None = None, epochs: int = config.SEARCH_EPOCHS,
             patience: int = config.SEARCH_PATIENCE, n_mc: int = config.SEARCH_N_MC,
             collect: list | None = None, arch: str = "hard", dip: bool = False) -> float:
    """fold 별 검증 AUC 평균. `collect` 를 주면 fold 별 물리 모수도 모은다(XAI: 모수 안정성)."""
    cols = columns(base_cols, refine)
    scores = []
    for k, fd in enumerate(folds):
        X = frame(X_dev, refs[fd["tag"]], refine)
        set_seed()
        D, var_t61, _, y, _ = T.tensors(X, cols, fd["tr"], dev, window=hp.get("window") if arch == "lstm" else None)
        net, _ = T.fit(train, D, var_t61, fd["tr"], fd["va"], hp, epochs=epochs,
                       patience=patience, n_mc=n_mc, dev=dev, log_every=None, arch=arch, dip=dip)
        p = T.predict_prob(net, D, fd["va"], dev, n_mc=64)
        ev = (y[fd["va"]] < config.FREEZE_C).astype(int)
        s = float(roc_auc_score(ev, p)) if 0 < ev.sum() < len(ev) else np.nan
        scores.append(s)
        if collect is not None:
            collect.append({"계열": train, "fold": fd["tag"], "검증AUC": round(s, 4),
                            "검증사건": int(ev.sum()), "μ_JT": net.mu_jt.item(),
                            "UA_kW/K": net.ua_kw.item(), "N₀": net.n0.item(),
                            "T_g_a": net.tg_a.item(), "T_g_b": net.tg_b.item()})
        if trial is not None:
            trial.set_user_attr(f"auc_{fd['tag']}", s)
            trial.report(float(np.nanmean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        del D, net
        if dev == "cuda":
            torch.cuda.empty_cache()
    return float(np.nanmean(scores))
