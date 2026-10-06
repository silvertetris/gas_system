"""데이터 효율 실험 — 학습 데이터가 적을 때 KF–soft PINN(가중 확장) vs 신경망(nn) vs GBM (2026-09-16).

설계 (계열 M · 타깃 min · 정제 KF):
  · 시험구간·조기종료/보정 구간은 **최종 모델과 똑같이 고정**한다(검증 = 개발 뒤 10%, 사건 28).
    검증 구간을 줄어든 창 끝에서 떼면 1~3년 창은 검증 사건이 0 이라 등장성 보정이 상수가 된다.
  · 학습 창만 줄인다: 검증 시작(퍼지 제외) 직전 **최근 1·2·3·5년 / 전체**. 시간 순서 유지.
  · 하이퍼파라미터는 전체 데이터 Optuna 최적값 고정(softw_min/v_KF · nn_min/v_KF). 창마다 재탐색하지 않는다.
  · 표준화 통계는 그 창의 학습 행만으로 계산한다(`T.tensors` 가 학습 마스크 사용) — 실제로 가진 데이터만 쓴다.
  · 신경망 두 구조는 시드 5회, GBM 은 random_state 고정 1회. 경보는 시험구간 상위 1% 같은 시간 수.
  · Brier 개선 기준은 창과 무관하게 **전체 학습 행 사건률**로 둔다(창끼리 비교 가능하게).
  · 전체 창 × softw 는 최종 모델 재현(시험 AUC ≈ 0.926 · 경보 74) — 검증용.

    .venv/bin/python -m pinn.restricted.data_efficiency            # 전체 (재시작하면 끝난 조합은 건너뜀)
    .venv/bin/python -m pinn.restricted.data_efficiency --years 1 --archs softw --seeds 42   # 일부만
산출: output/optuna/data_efficiency/metrics.csv · run.log
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, folds, train as T, tune

log = logging.getLogger("pinn.restricted.data_efficiency")
OUT = config.OPTUNA_DIR / "data_efficiency"
BEST = {"softw": config.OPTUNA_DIR / "softw_min" / "v_KF" / "best_M.json",
        "nn": config.OPTUNA_DIR / "nn_min" / "v_KF" / "best_M.json"}
REFINE = ("kf",)
SEEDS = (42, 7, 123, 2024, 31337)


def hits_top1(p: np.ndarray, ev: np.ndarray, tie: np.ndarray) -> int:
    k = int(np.ceil(0.01 * len(p)))
    top = np.zeros(len(p), bool)
    top[np.lexsort((tie, -p))[:k]] = True
    return int((top & ev).sum())


def run(train: str, target: str, years_list, archs, seeds, dev: str, save_pred: bool = False) -> pd.DataFrame:
    """`save_pred=True`: 끝난 조합도 다시 학습해 시험 예측(p·mu·sd·y·ev)을 parquet 로 저장하고, metrics.csv 는 건드리지 않는다."""
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "metrics.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["years", "model", "seed"])
    key = set(zip(done["years"].astype(str), done["model"], done["seed"].astype(str)))

    Xb, base_cols = folds.base_frame(train, target)
    dev_m, test_m, t_cut = folds.dev_test(Xb)
    in_tr, in_va, _ = folds.inner_split(Xb[dev_m])
    ref = folds.load_refinements(train, ["final"])["final"]
    cols = tune.columns(base_cols, REFINE)
    X = tune.frame(Xb, ref, REFINE)
    pos = np.flatnonzero(dev_m)
    tr_full = np.zeros(len(X), bool); tr_full[pos[in_tr]] = True
    va = np.zeros(len(X), bool); va[pos[in_va]] = True
    ev = (X["y_t61_min"] < config.FREEZE_C).to_numpy(float)
    ev_te = ev[test_m] > 0.5
    base_full = float(ev[tr_full].mean())
    ite = torch.tensor(np.flatnonzero(test_m), device=dev)
    tr_end = X.index[tr_full][-1]
    log.info("[%s] 시험 %s~ (%d행, 사건 %d) | 검증 %s~%s (%d행, 사건 %d) | 전체 학습 끝 %s", train, t_cut,
             int(test_m.sum()), int(ev_te.sum()), X.index[va][0], X.index[va][-1], int(va.sum()),
             int(ev[va].sum()), tr_end)

    def save(row):
        nonlocal done
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(out_csv, index=False)

    for years in years_list:
        tr = tr_full.copy()
        if years is not None:
            tr &= np.asarray(X.index > tr_end - pd.DateOffset(years=years))
        ytag = "all" if years is None else str(years)
        n_ev = int(ev[tr].sum())
        span = f"{X.index[tr][0].date()}~{X.index[tr][-1].date()}"
        log.info("[%s] 학습 창 %s년: %s · %d행 · 사건 %d", train, ytag, span, int(tr.sum()), n_ev)
        D, var_t61, _, y, _ = T.tensors(X, cols, tr, dev)
        common = {"years": ytag, "train_span": span, "train_rows": int(tr.sum()), "train_events": n_ev,
                  "test_rows": int(test_m.sum()), "test_events": int(ev_te.sum())}

        def keep_pred(model, seed, p, mu=None, sd=None, m=None, h=None):
            pd.DataFrame({"p": p, "mu": mu if mu is not None else np.nan, "sd": sd if sd is not None else np.nan,
                          "y": y[test_m], "ev": ev_te.astype(float)}, index=X.index[test_m]
                         ).to_parquet(OUT / f"pred_{ytag}_{model}_{seed}.parquet")
            log.info("[%s|%s년] %-5s 시드 %5d 예측 저장 — AUC %.3f · 경보 %d", train, ytag, model, seed, m["AUC"], h)

        if "gbm" in archs and (save_pred or (ytag, "gbm", str(config.RANDOM_SEED)) not in key):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6,
                                                 l2_regularization=1.0, random_state=config.RANDOM_SEED,
                                                 early_stopping=True, validation_fraction=0.15)
            clf.fit(X.loc[tr, cols], ev[tr])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                clf.predict_proba(X.loc[va, cols])[:, 1], ev[va])
            p = np.clip(iso.predict(clf.predict_proba(X.loc[test_m, cols])[:, 1]), 0, 1)
            m = T.metrics(p, ev_te.astype(float), base_full)
            h = hits_top1(p, ev_te, np.random.default_rng(config.RANDOM_SEED).random(len(p)))
            if save_pred:
                keep_pred("gbm", config.RANDOM_SEED, p, m=m, h=h)
            else:
                save({**common, "model": "gbm", "seed": config.RANDOM_SEED, **m, "alarm_hits": h, "rmse": np.nan})
                log.info("[%s|%s년] GBM  AUC %.3f · 경보 %d", train, ytag, m["AUC"], h)

        for arch in [a for a in ("softw", "nn") if a in archs]:
            hp = json.loads(BEST[arch].read_text(encoding="utf-8"))["hp"]
            dip = arch == "softw" and target != "mean"
            for seed in seeds:
                if not save_pred and (ytag, arch, str(seed)) in key:
                    continue
                tune.set_seed(seed)
                net, best = T.fit(train, D, var_t61, tr, va, hp, epochs=config.FINAL_EPOCHS,
                                  patience=config.FINAL_PATIENCE, n_mc=config.N_MC, dev=dev, log_every=None,
                                  arch=arch, dip=dip)
                p_va = T.predict_prob(net, D, va, dev)
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_va, ev[va])
                p = np.clip(iso.predict(T.predict_prob(net, D, test_m, dev)), 0, 1)
                mu, sd = T.predict_mean_sd(net, D, ite)
                m = T.metrics(p, ev_te.astype(float), base_full)
                h = hits_top1(p, ev_te, mu)
                rmse = float(np.sqrt(np.mean((mu - y[test_m]) ** 2)))
                row = {**common, "model": arch, "seed": seed, **m, "alarm_hits": h, "rmse": round(rmse, 3),
                       "best_valid_loss": round(float(best), 4)}
                if save_pred:
                    keep_pred(arch, seed, p, mu, sd, m, h)
                    del net
                    continue
                if arch == "softw":
                    row.update({"mu_jt": round(net.mu_jt.item(), 4), "ua_kw": round(net.ua_kw.item(), 3),
                                "n0": round(net.n0.item(), 3)})
                save(row)
                log.info("[%s|%s년] %-5s 시드 %5d  AUC %.3f · 경보 %d · RMSE %.2f", train, ytag, arch, seed,
                         m["AUC"], h, rmse)
                del net
                if dev == "cuda":
                    torch.cuda.empty_cache()
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="M")
    ap.add_argument("--target", default="min")
    ap.add_argument("--years", nargs="+", default=["1", "2", "3", "5", "all"])
    ap.add_argument("--archs", nargs="+", default=["gbm", "softw", "nn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-pred", action="store_true", help="retrain given combos and save test predictions only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "run.log", encoding="utf-8")])
    years = [None if y == "all" else int(y) for y in a.years]
    d = run(a.train, a.target, years, a.archs, a.seeds, a.dev, save_pred=a.save_pred)
    if a.save_pred:
        return
    s = d.groupby(["years", "model"])[["AUC", "alarm_hits", "rmse"]].agg(["mean", "std"]).round(3)
    log.info("요약\n%s", s.to_string())


if __name__ == "__main__":
    main()
