"""Kalman Filter 파일럿 진입점 — PINN 이전 정제 단계.

    .venv/bin/python -m kalman.test.pipeline
    .venv/bin/python -m kalman.test.pipeline --reuse-study

흐름 (lstm/AE 와 동일 절차):
    표본 로드 → 시계열 구성 → 시간순 8:2 분할(+purge)
    → Optuna(TimeSeriesSplit 5-fold)로 Q/R 탐색  [학습 구간만]
    → 최적 Q/R 로 테스트 구간 필터링
    → 단순 평활기 베이스라인 대조 → 신호별 진단 → 플롯

출력(kalman/test/output/):
    metrics.csv          KF vs 베이스라인 1스텝 예측 성능
    diagnostics.csv      신호별 Kalman gain / innovation / NIS
    filtered.csv         정제된 신호 궤적           → PINN 입력
    outlier_flags.csv    NIS 기준 이상 시각          → PINN 학습 제외 후보
    optuna_trials.csv / figures/
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from . import baseline, config, data, explain, model, plots, tune

logger = logging.getLogger("kf.test")


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
    params["q_slope"] = 0.0 if pd.isna(params.get("q_slope")) else float(params["q_slope"])
    logger.info("이전 탐색 재사용: CV %.5f | %s", best["value"], params)
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description="KF 파일럿")
    ap.add_argument("--reuse-study", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. 표본 → 시계열 → 분할 -------------------------------------------
    df = data.load_sample()
    Z_raw, t = data.to_series(df)
    tr_idx, te_idx = data.split_purged(len(Z_raw))
    logger.info("학습 %s ~ %s / 테스트 %s ~ %s",
                t[tr_idx[0]], t[tr_idx[-1]], t[te_idx[0]], t[te_idx[-1]])

    # --- 2. Optuna 로 Q/R 탐색 (학습 구간만) --------------------------------
    best = load_best_params() if args.reuse_study else None
    if best is None:
        logger.info("=== Optuna 탐색 (%d trials × %d-fold) ===", config.N_TRIALS, config.N_SPLITS)
        study = tune.run_search(Z_raw[tr_idx])
        study.trials_dataframe().to_csv(config.OUTPUT_DIR / "optuna_trials.csv", index=False)
        best = dict(study.best_params)
        best.setdefault("q_slope", 0.0)

    # --- 3. 테스트 구간 필터링 ----------------------------------------------
    # 스케일러는 학습 구간으로만 fit. 필터는 인과라 학습부터 이어서 돌리되(초기 과도 제거)
    # 평가는 테스트 구간에서만 한다.
    xs = data.Scaler().fit(Z_raw[tr_idx][None, ...])
    run_idx = np.arange(tr_idx[0], te_idx[-1] + 1)
    Zs = xs.transform(Z_raw[run_idx][None, ...])[0]
    res = model.run_filter(Zs, best, dt=config.DT)
    local_te = te_idx - run_idx[0]
    res_te = {k: v[local_te] for k, v in res.items()}
    Zs_te = Zs[local_te]

    kf_rmse, kf_per = model.one_step_rmse(Zs_te, res_te["predicted"])

    # --- 4. 베이스라인 -------------------------------------------------------
    rows = [{"model": f"KF({best['model_type']})", "rmse": kf_rmse}]
    rows.append({"model": "persistence", "rmse": model.one_step_rmse(
        Zs_te, baseline.persistence(Zs_te))[0]})
    for a in config.EWMA_ALPHAS:
        rows.append({"model": f"EWMA(α={a})", "rmse": model.one_step_rmse(
            Zs_te, baseline.ewma(Zs_te, a))[0]})
    for w in config.MA_WINDOWS:
        rows.append({"model": f"MA({w})", "rmse": model.one_step_rmse(
            Zs_te, baseline.moving_average(Zs_te, w))[0]})
    met = pd.DataFrame(rows)
    ref = met.loc[met["model"] == "persistence", "rmse"].iloc[0]
    met["vs_persistence_%"] = (100 * (ref - met["rmse"]) / ref).round(2)
    met.to_csv(config.OUTPUT_DIR / "metrics.csv", index=False)
    logger.info("1스텝 예측 성능:\n%s", met.sort_values("rmse").to_string(index=False))

    # --- 5. 신호별 진단 / 정제 산출물 ---------------------------------------
    diag = explain.signal_diagnostics(Zs_te, res_te, config.SIGNALS, xs.scale_)
    diag.to_csv(config.OUTPUT_DIR / "diagnostics.csv", index=False)
    logger.info("신호별 진단:\n%s", diag.to_string(index=False))

    filt_raw = res_te["filtered"] * xs.scale_ + xs.center_
    pd.DataFrame(filt_raw, columns=config.SIGNALS, index=t[te_idx]).to_csv(
        config.OUTPUT_DIR / "filtered.csv")
    explain.outlier_flags(res_te, config.SIGNALS, t[te_idx]).to_csv(
        config.OUTPUT_DIR / "outlier_flags.csv", index=False)

    # --- 6. 그림 -------------------------------------------------------------
    plots.plot_vs_baselines(met)
    plots.plot_filtering(t[te_idx], Z_raw[te_idx], 
                         {"filtered": filt_raw}, config.SIGNALS)
    plots.plot_diagnostics(diag)
    plots.plot_innovation_timeline(t[te_idx], res_te, config.SIGNALS)
    logger.info("=== 완료: %s ===", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
