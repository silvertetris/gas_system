"""LSTM 파일럿 진입점.

    .venv/bin/python -m lstm.test.pipeline

흐름:
    표본 로드 → 윈도우 생성 → 시간순 8:2 분할(+purge)
    → Optuna(TimeSeriesSplit 5-fold)로 하이퍼파라미터 탐색   [학습 구간만 사용]
    → 최적 파라미터로 최종 재학습 → 테스트 예측
    → 베이스라인 대조 + SHAP 변수 중요도

출력(lstm/test/output/):
    predictions.csv        테스트 구간 실측/예측/잔차
    metrics.csv            LSTM vs 베이스라인 성능
    optuna_trials.csv      탐색 이력
    shap_importance.csv    피처별 SHAP 중요도
    shap_time_profile.csv  윈도우 내 시간축 프로파일
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch

from . import config, data, explain, model, tune

logger = logging.getLogger("lstm.test")


def baselines(y_true: np.ndarray, X_test_raw: np.ndarray, feat_idx: dict) -> dict:
    """LSTM 이 반드시 이겨야 할 무학습 기준선들.

    - persistence : T_out(t+H) = T_out(t)              (윈도우 마지막 시점의 TI33P)
    - physics_eps : T_out = T_in + ε̄·(T_bath − T_in)   (ε̄ = 학습구간 중앙 ε, 인자로 받음)
    """
    last = X_test_raw[:, -1, :]
    return {
        "persistence": last[:, feat_idx["TI33P"]],
    }


def physics_baseline(X_raw: np.ndarray, feat_idx: dict, eps_bar: float) -> np.ndarray:
    last = X_raw[:, -1, :]
    t_in, t_bath = last[:, feat_idx["TI21Z"]], last[:, feat_idx["TI-D2P"]]
    return t_in + eps_bar * (t_bath - t_in)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def load_best_params() -> dict | None:
    """이전 탐색 결과를 재사용한다 (`--reuse-study`). Optuna 재실행(수십 분)을 건너뛰기 위함."""
    p = config.OUTPUT_DIR / "optuna_trials.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["state"] == "COMPLETE"]
    if df.empty:
        return None
    best = df.loc[df["value"].idxmin()]
    params = {c.removeprefix("params_"): best[c] for c in df.columns if c.startswith("params_")}
    for k in ("hidden_size", "num_layers", "batch_size"):
        params[k] = int(params[k])
    logger.info("이전 탐색 재사용: CV RMSE %.4f ℃ | %s", best["value"], params)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM 파일럿")
    parser.add_argument("--reuse-study", action="store_true",
                        help="기존 optuna_trials.csv 의 최적 파라미터를 재사용(탐색 건너뜀)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.set_seed()
    dev = model.get_device()
    logger.info("device: %s", dev)

    # --- 1. 표본 → 윈도우 --------------------------------------------------
    df = data.load_sample()
    X, y, t_start, t_target = data.make_windows(df)

    # --- 2. 시간순 8:2 분할 (+ purge) ---------------------------------------
    tr_idx, te_idx = data.split_purged(len(X))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    logger.info("학습 %s ~ %s / 테스트 %s ~ %s",
                t_target[tr_idx[0]], t_target[tr_idx[-1]],
                t_target[te_idx[0]], t_target[te_idx[-1]])

    # --- 3. Optuna 탐색 (학습 구간만) ---------------------------------------
    best = load_best_params() if args.reuse_study else None
    if best is None:
        logger.info("=== Optuna 탐색 (%d trials × %d-fold TimeSeriesSplit) ===",
                    config.N_TRIALS, config.N_SPLITS)
        study = tune.run_search(X_tr, y_tr)
        study.trials_dataframe().to_csv(config.OUTPUT_DIR / "optuna_trials.csv", index=False)
        best = study.best_params

    # --- 4. 최종 재학습 -----------------------------------------------------
    # 스케일러는 여기서도 **학습 구간으로만** fit 한다.
    xs, ys = data.Scaler().fit(X_tr), data.TargetScaler().fit(y_tr)
    # 최종 학습의 early stopping 용 검증셋은 학습 구간의 마지막 10%(시간순, purge 적용)
    cut = int(len(X_tr) * 0.9)
    inner_tr = np.arange(0, max(cut - config.PURGE, 1))
    inner_va = np.arange(cut, len(X_tr))
    logger.info("=== 최종 학습 (inner train %d / val %d) ===", len(inner_tr), len(inner_va))
    net, _ = model.train_model(
        xs.transform(X_tr[inner_tr]), ys.transform(y_tr[inner_tr]),
        xs.transform(X_tr[inner_va]), ys.transform(y_tr[inner_va]),
        best, epochs=config.FINAL_EPOCHS, patience=config.EARLY_STOP_PATIENCE,
        device=dev, verbose=True,
    )

    torch.save({"state_dict": net.state_dict(), "params": best,
                "scaler": {"center": xs.center_, "scale": xs.scale_},
                "target_scaler": {"center": ys.center_, "scale": ys.scale_}},
               config.OUTPUT_DIR / "model.pt")

    # --- 5. 테스트 예측 + 베이스라인 ----------------------------------------
    pred = ys.inverse(model.predict(net, xs.transform(X_te), device=dev))
    fidx = {f: i for i, f in enumerate(config.FEATURES)}

    # ε̄ 는 **학습 구간에서만** 계산한다(누수 방지)
    last_tr = X_tr[:, -1, :]
    denom = last_tr[:, fidx["TI-D2P"]] - last_tr[:, fidx["TI21Z"]]
    num = last_tr[:, fidx["TI33P"]] - last_tr[:, fidx["TI21Z"]]
    eps_bar = float(np.median(num[denom > 15] / denom[denom > 15]))
    logger.info("학습구간 ε̄ = %.3f", eps_bar)

    preds = {"lstm": pred, **baselines(y_te, X_te, fidx),
             "physics_eps": physics_baseline(X_te, fidx, eps_bar)}
    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "RMSE": rmse(y_te, p), "MAE": mae(y_te, p)})
    met = pd.DataFrame(rows)
    base = met.loc[met["model"] == "persistence", "RMSE"].iloc[0]
    met["vs_persistence_%"] = (100 * (base - met["RMSE"]) / base).round(2)
    met.to_csv(config.OUTPUT_DIR / "metrics.csv", index=False)
    logger.info("성능:\n%s", met.to_string(index=False))

    pd.DataFrame({"target_time": t_target[te_idx], "actual": y_te, **preds,
                  "residual_lstm": y_te - pred}).to_csv(
        config.OUTPUT_DIR / "predictions.csv", index=False)

    # --- 6. SHAP ------------------------------------------------------------
    logger.info("=== SHAP (배경 %d / 설명 %d) ===", config.SHAP_BACKGROUND, config.SHAP_EXPLAIN)
    rng = np.random.default_rng(config.SEED)
    bg = xs.transform(X_tr[rng.choice(len(X_tr), min(config.SHAP_BACKGROUND, len(X_tr)), replace=False)])
    ex = xs.transform(X_te[rng.choice(len(X_te), min(config.SHAP_EXPLAIN, len(X_te)), replace=False)])
    imp, sv = explain.shap_importance(net, bg, ex, config.FEATURES)
    imp.to_csv(config.OUTPUT_DIR / "shap_importance.csv", index=False)
    explain.time_profile(sv, config.FEATURES).to_csv(config.OUTPUT_DIR / "shap_time_profile.csv")
    logger.info("SHAP 중요도:\n%s", imp.to_string(index=False))
    logger.info("=== 완료: %s ===", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
