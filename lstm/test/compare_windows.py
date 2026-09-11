"""입력 윈도우 길이 비교 — 긴 윈도우가 실제로 쓸모 있나.

배경: 180분 윈도우로 학습한 모델의 SHAP 시간축 프로파일을 보니
**최근 30분이 기여의 83.6%**, 가장 오래된 30분은 0.4%였다. 두 해석이 가능하다.
  (a) 60분 지평에는 단기 관성만으로 충분하다 → 윈도우를 줄여도 성능이 같다
  (b) 모델이 장기 의존을 **학습하지 못했다**  → 윈도우를 줄이면 성능이 나빠진다
이 스크립트가 둘을 가른다. (a)면 윈도우를 줄여 학습을 몇 배 빠르게 할 수 있고,
(b)면 수조 시정수(63분)의 느린 동역학을 LSTM 이 못 잡는다는 뜻이라 KF-PINN 대비 구조적 열세다.

⚠ 공정성: **하이퍼파라미터는 180분 탐색 결과로 고정**한다. 윈도우 길이만 바꿔 그 효과를 격리한다.
   (윈도우별 재탐색이 더 엄밀하지만 파일럿 비용이 3배가 된다 — 결론이 애매하면 그때 하면 된다.)

실행: .venv/bin/python -m lstm.test.compare_windows
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import config, data, model, plots
from .pipeline import load_best_params, mae, rmse

logger = logging.getLogger("lstm.test.compare")

WINDOWS = [15, 30, 60, 120, 180]


def run_one(df: pd.DataFrame, window: int, params: dict) -> dict:
    X, y, _, t_target = data.make_windows(df, window=window)
    purge = window + config.HORIZON
    tr_idx, te_idx = data.split_purged(len(X), purge=purge)
    X_tr, y_tr, X_te, y_te = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]

    xs, ys = data.Scaler().fit(X_tr), data.TargetScaler().fit(y_tr)
    cut = int(len(X_tr) * 0.9)
    inner_tr = np.arange(0, max(cut - purge, 1))
    inner_va = np.arange(cut, len(X_tr))

    t0 = time.time()
    net, _ = model.train_model(
        xs.transform(X_tr[inner_tr]), ys.transform(y_tr[inner_tr]),
        xs.transform(X_tr[inner_va]), ys.transform(y_tr[inner_va]),
        params, epochs=config.FINAL_EPOCHS, patience=config.EARLY_STOP_PATIENCE,
    )
    train_sec = time.time() - t0
    pred = ys.inverse(model.predict(net, xs.transform(X_te)))

    fidx = {f: i for i, f in enumerate(config.FEATURES)}
    persist = X_te[:, -1, fidx["TI33P"]]
    out = {
        "window": window, "n_train": len(X_tr), "n_test": len(X_te),
        "RMSE": rmse(y_te, pred), "MAE": mae(y_te, pred),
        "persistence_RMSE": rmse(y_te, persist), "train_sec": round(train_sec, 1),
    }
    out["vs_persistence_%"] = round(100 * (out["persistence_RMSE"] - out["RMSE"]) / out["persistence_RMSE"], 2)
    logger.info("window %3d분 → RMSE %.3f ℃ (persistence %.3f, %+.1f%%) | %.0fs",
                window, out["RMSE"], out["persistence_RMSE"], out["vs_persistence_%"], train_sec)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model.set_seed()
    params = load_best_params()
    if params is None:
        raise RuntimeError("optuna_trials.csv 없음 — 먼저 `python -m lstm.test.pipeline` 실행 필요.")

    df = data.load_sample()
    rows = [run_one(df, w, params) for w in WINDOWS]
    res = pd.DataFrame(rows)
    res.to_csv(config.OUTPUT_DIR / "window_comparison.csv", index=False)
    print("\n" + res.to_string(index=False))
    plots.plot_window_comparison(res)

    best = res.loc[res["RMSE"].idxmin()]
    ref = res.loc[res["window"] == 180, "RMSE"].iloc[0]
    logger.info("최적 윈도우 %d분 (RMSE %.3f). 180분 대비 %+.2f%%",
                int(best["window"]), best["RMSE"], 100 * (best["RMSE"] - ref) / ref)


if __name__ == "__main__":
    main()
