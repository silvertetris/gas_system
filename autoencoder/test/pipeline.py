"""Autoencoder 파일럿 진입점 — PINN 이전 정제 단계.

    .venv/bin/python -m autoencoder.test.pipeline
    .venv/bin/python -m autoencoder.test.pipeline --reuse-study   # Optuna 건너뜀

흐름 (lstm.test.pipeline 과 동일 절차):
    표본 로드 → 윈도우 → 시간순 8:2 분할(+purge)
    → Optuna(TimeSeriesSplit 5-fold)  [학습 구간만]
    → 최종 재학습 → 테스트 재구성
    → PCA 베이스라인 대조 → 피처별 오차 → SHAP → 플롯

출력(autoencoder/test/output/):
    metrics.csv            AE vs PCA 재구성 성능
    feature_error.csv      피처별 재구성 난이도
    recon_error.csv        윈도우별 재구성오차 (PINN 학습 제외구간 판정용)
    latent.npy             잠재벡터 (PINN 상태표현 후보)
    shap_importance.csv    재구성오차에 대한 SHAP
    optuna_trials.csv / model.pt / figures/
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch

from . import baseline, config, data, explain, model, plots, tune

logger = logging.getLogger("ae.test")


def load_best_params() -> dict | None:
    p = config.OUTPUT_DIR / "optuna_trials.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["state"] == "COMPLETE"]
    if df.empty:
        return None
    best = df.loc[df["value"].idxmin()]
    params = {c.removeprefix("params_"): best[c] for c in df.columns if c.startswith("params_")}
    for k in ("latent_dim", "hidden_size", "num_layers", "batch_size"):
        params[k] = int(params[k])
    logger.info("이전 탐색 재사용: CV %.4f | %s", best["value"], params)
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description="AE 파일럿")
    ap.add_argument("--reuse-study", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.set_seed(config.SEED)
    dev = model.get_device()
    logger.info("device: %s", dev)

    # --- 1. 표본 → 윈도우 → 분할 -------------------------------------------
    df = data.load_sample()
    X, t_start = data.make_windows(df)
    tr_idx, te_idx = data.split_purged(len(X))
    X_tr, X_te = X[tr_idx], X[te_idx]
    logger.info("학습 %s ~ %s / 테스트 %s ~ %s",
                t_start[tr_idx[0]], t_start[tr_idx[-1]],
                t_start[te_idx[0]], t_start[te_idx[-1]])

    # --- 2. Optuna (학습 구간만) -------------------------------------------
    best = load_best_params() if args.reuse_study else None
    if best is None:
        logger.info("=== Optuna 탐색 (%d trials × %d-fold) ===", config.N_TRIALS, config.N_SPLITS)
        study = tune.run_search(X_tr)
        study.trials_dataframe().to_csv(config.OUTPUT_DIR / "optuna_trials.csv", index=False)
        best = study.best_params

    # --- 3. 최종 재학습 (스케일러는 학습 구간으로만 fit) ---------------------
    xs = data.Scaler().fit(X_tr)
    cut = int(len(X_tr) * 0.9)
    inner_tr = np.arange(0, max(cut - config.PURGE, 1))
    inner_va = np.arange(cut, len(X_tr))
    logger.info("=== 최종 학습 (inner train %d / val %d) ===", len(inner_tr), len(inner_va))
    ae, _ = model.train_model(xs.transform(X_tr[inner_tr]), xs.transform(X_tr[inner_va]),
                              best, epochs=config.FINAL_EPOCHS,
                              patience=config.EARLY_STOP_PATIENCE, device=dev, verbose=True)
    torch.save({"state_dict": ae.state_dict(), "params": best,
                "scaler": {"center": xs.center_, "scale": xs.scale_}},
               config.OUTPUT_DIR / "model.pt")

    # --- 4. 테스트 재구성 + PCA 베이스라인 ----------------------------------
    Xs_tr, Xs_te = xs.transform(X_tr), xs.transform(X_te)
    rec_s = model.reconstruct(ae, Xs_te, device=dev)
    rec_raw = xs.inverse(rec_s)
    # ⚠ 채점은 **학습이 최적화한 공간(스케일 후)** 에서 한다. 원 단위로만 재면
    #   분산이 큰 신호(PI-D2P σ17)가 지표를 지배해 비교가 왜곡된다.
    ae_rmse = float(np.sqrt(np.mean((rec_s - Xs_te) ** 2)))
    ae_rmse_raw = float(np.sqrt(np.mean((rec_raw - X_te) ** 2)))

    pca_rows = []
    for k in config.PCA_COMPONENTS:
        # PCA 도 **같은 스케일 공간**에서 돌려야 공정하다. 1차 실행에서는 PCA 만
        # 원 공간에서 돌려서 채점 지표를 그대로 최적화하는 이점을 줬다(불공정).
        r, rec_p = baseline.pca_reconstruction_rmse(Xs_tr, Xs_te, k)
        r_raw = float(np.sqrt(np.mean((xs.inverse(rec_p) - X_te) ** 2)))
        pca_rows.append({"model": f"PCA(k={k})", "n_components": k,
                         "rmse": r, "rmse_raw": r_raw})
    pca = pd.DataFrame(pca_rows)
    met = pd.concat([
        pd.DataFrame([{"model": f"LSTM-AE(latent={best['latent_dim']})",
                       "n_components": best["latent_dim"],
                       "rmse": ae_rmse, "rmse_raw": ae_rmse_raw}]), pca,
    ], ignore_index=True)
    same = pca.loc[pca["n_components"] == best["latent_dim"], "rmse"]
    if len(same):
        met["vs_same_dim_PCA_%"] = (100 * (same.iloc[0] - met["rmse"]) / same.iloc[0]).round(2)
    met.to_csv(config.OUTPUT_DIR / "metrics.csv", index=False)
    logger.info("재구성 성능:\n%s", met.to_string(index=False))

    # --- 5. 피처별 오차 / 오차 시계열 / 잠재벡터 ----------------------------
    ferr = explain.per_feature_error(Xs_te, rec_s, config.FEATURES, scale=xs.scale_)
    ferr.to_csv(config.OUTPUT_DIR / "feature_error.csv", index=False)
    logger.info("피처별 재구성 난이도:\n%s", ferr.to_string(index=False))

    win_err = ((rec_s - Xs_te) ** 2).mean(axis=(1, 2))
    pd.DataFrame({"window_start": t_start[te_idx], "recon_mse": win_err}).to_csv(
        config.OUTPUT_DIR / "recon_error.csv", index=False)

    z = model.encode_all(ae, Xs_te, device=dev)
    np.save(config.OUTPUT_DIR / "latent.npy", z)
    logger.info("잠재벡터 %s 저장 (PINN 상태표현 후보)", z.shape)

    # --- 6. SHAP ------------------------------------------------------------
    rng = np.random.default_rng(config.SEED)
    bg = xs.transform(X_tr[rng.choice(len(X_tr), min(config.SHAP_BACKGROUND, len(X_tr)), replace=False)])
    ex = Xs_te[rng.choice(len(Xs_te), min(config.SHAP_EXPLAIN, len(Xs_te)), replace=False)]
    imp, _ = explain.shap_on_error(ae, bg, ex, config.FEATURES)
    imp.to_csv(config.OUTPUT_DIR / "shap_importance.csv", index=False)
    logger.info("SHAP(재구성오차):\n%s", imp.to_string(index=False))

    # --- 7. 그림 ------------------------------------------------------------
    plots.plot_vs_pca(ae_rmse, pca, best["latent_dim"])
    plots.plot_feature_error(ferr)
    plots.plot_reconstruction(Xs_te, rec_s, config.FEATURES)
    plots.plot_error_timeline(t_start[te_idx], win_err)
    plots.plot_latent(z, t_start[te_idx])
    plots.plot_shap(imp)
    logger.info("=== 완료: %s ===", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
