"""제한 PINN · 정제 데이터 · Optuna · 8:2 시간분할 CV — 진입점.

    .venv/bin/python -m pinn.restricted.optuna_pipeline            # 본 실행
    .venv/bin/python -m pinn.restricted.optuna_pipeline --smoke    # 배선 점검(2 trial × 2 fold × 3 epoch)

흐름 (`lstm/test/pipeline.py` 와 같은 틀):
    기반 특징 → 시간순 8:2 (+퍼지) → 개발구간 TimeSeriesSplit 5-fold
    → fold 경계별 AE·EKF 재적합 (KF 는 고정 이득)
    → Optuna: 정제 조합 + 하이퍼파라미터, 목적 = fold 검증 AUC 평균   [개발구간만]
    → 최적값을 fold 별로 다시 돌려 물리 모수 안정성 기록
    → 최종 학습: 개발구간 앞 90% 학습 · 뒤 10% 조기종료·등장성 보정
    → 시험구간 20% 1회 평가 + 같은 입력·같은 분할 GBM 기준선
    → XAI 용 산출(모델·입력행렬·이름) 저장
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, folds, train as T, tune

log = logging.getLogger("pinn.restricted.optuna")


def run_train(train: str, out_dir, smoke: bool, dev: str, prep_only: bool = False,
              variant: str = "auto", seeds=(config.RANDOM_SEED,), arch: str = "hard",
              target: str = "min_raw") -> list[dict]:
    """`variant="auto"` 면 정제 조합도 Optuna 가 고른다. 조합 이름을 주면 그 조합으로 **고정**하고
    하이퍼파라미터만 찾은 뒤, 최종 학습을 `seeds` 마다 반복한다(조합 간 짝비교용)."""
    fixed = None if variant == "auto" else variant
    dip = arch in ("soft", "softw") and target != "mean"   # 분 단위 최저 타깃일 때만 순간하강 헤드
    Xb, base_cols = folds.base_frame(train, target)
    log.info("[%s] 구조 %s · 타깃 %s · 정제 %s · 순간하강 헤드 %s", train, arch, target, variant, dip)
    dev_m, test_m, t_cut = folds.dev_test(Xb)
    X_dev = Xb[dev_m]
    fl = folds.cv_folds(X_dev, n_splits=config.N_SPLITS)
    if smoke:
        fl = fl[:2]
    in_tr, in_va, in_fit_end = folds.inner_split(X_dev)
    fit_ends = {fd["tag"]: fd["fit_end"] for fd in fl} | {"final": in_fit_end}
    ev_all = (Xb["y_t61_min"] < config.FREEZE_C).to_numpy()
    log.info("[%s] 개발 %s~%s (%d행, 사건 %d) | 시험 %s~ (%d행, 사건 %d)", train,
             X_dev.index[0].date(), X_dev.index[-1].date(), len(X_dev), int(ev_all[dev_m].sum()),
             t_cut.date(), int(test_m.sum()), int(ev_all[test_m].sum()))
    for fd in fl:
        log.info("   %s 학습 ~%s (%d) · 검증 %s~%s (%d, 사건 %d)", fd["tag"], X_dev.index[fd["tr"]][-1].date(),
                 int(fd["tr"].sum()), X_dev.index[fd["va"]][0].date(), X_dev.index[fd["va"]][-1].date(),
                 int(fd["va"].sum()), int(ev_all[dev_m][fd["va"]].sum()))
    if not smoke:
        folds.ensure_refinements(train, fit_ends)
    if prep_only:                                  # 정제 산출물만 미리 만든다 (GPU 대기 중 CPU 작업)
        return [{"계열": train, "준비": "정제 산출 완료"}]
    refs = folds.load_refinements(train, list(fit_ends), smoke=smoke)

    # --- Optuna (개발구간만)
    ep, pa = (3, 1) if smoke else (config.SEARCH_EPOCHS, config.SEARCH_PATIENCE)
    study = optuna.create_study(direction="maximize", study_name=f"pinn_{train}_{variant}_{arch}_{target}",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))

    def objective(trial):
        refine, hp = tune.suggest(trial, fixed, arch)
        s = tune.cv_score(train, X_dev, base_cols, refs, fl, refine, hp, dev, trial=trial, arch=arch, dip=dip,
                          epochs=ep, patience=pa)
        log.info("[%s] trial %2d  CV AUC %.4f  정제=%s  %s", train, trial.number, s,
                 trial.params.get("refine", fixed), {k: (round(v, 5) if isinstance(v, float) else v)
                                          for k, v in hp.items()})
        return s

    study.optimize(objective, n_trials=2 if smoke else config.N_TRIALS, show_progress_bar=False)
    trials = study.trials_dataframe()
    trials.to_csv(out_dir / f"trials_{train}.csv", index=False, encoding="utf-8-sig")
    refine, hp = tune.params_to_hp(study.best_params, fixed)
    refine_name = study.best_params.get("refine", fixed)
    log.info("[%s] 최적 CV AUC %.4f | 정제=%s | %s", train, study.best_value, refine_name, hp)
    try:
        imp = optuna.importance.get_param_importances(study)
        pd.Series(imp, name="중요도").to_csv(out_dir / f"param_importance_{train}.csv", encoding="utf-8-sig")
    except Exception as e:                       # 완료 trial 이 적으면 실패할 수 있다
        log.warning("[%s] 하이퍼파라미터 중요도 계산 생략: %s", train, e)

    # --- 최적값 fold 별 재실행 → 물리 모수 안정성 (XAI)
    fold_params: list = []
    tune.cv_score(train, X_dev, base_cols, refs, fl, refine, hp, dev, epochs=ep, patience=pa,
                  collect=fold_params, arch=arch, dip=dip)
    pd.DataFrame(fold_params).to_csv(out_dir / f"fold_params_{train}.csv", index=False, encoding="utf-8-sig")

    # --- 최종 학습: 개발 앞 90% / 뒤 10% (정제는 final 경계 = 앞 90% 끝까지 적합)
    cols = tune.columns(base_cols, refine)
    X = tune.frame(Xb, refs["final"], refine)
    pos = np.flatnonzero(dev_m)
    tr_all = np.zeros(len(X), bool); tr_all[pos[in_tr]] = True
    va_all = np.zeros(len(X), bool); va_all[pos[in_va]] = True
    D, var_t61, names, y, wp = T.tensors(X, cols, tr_all, dev,      # 난수 안 씀 → 시드와 무관
                                         window=hp.get("window") if arch == "lstm" else None)
    ev = (y < config.FREEZE_C).astype(float)
    fe, fp = (3, 1) if smoke else (config.FINAL_EPOCHS, config.FINAL_PATIENCE)

    # GBM 기준선 — 같은 입력·같은 분할. random_state 고정이라 시드 반복이 필요 없다
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6,
                                         l2_regularization=1.0, random_state=config.RANDOM_SEED,
                                         early_stopping=True, validation_fraction=0.15)
    clf.fit(X.loc[tr_all, cols], ev[tr_all])
    iso_g = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        clf.predict_proba(X.loc[va_all, cols])[:, 1], ev[va_all])
    g_te = np.clip(iso_g.predict(clf.predict_proba(X.loc[test_m, cols])[:, 1]), 0, 1)

    base = float(ev[tr_all].mean())
    obs_te = X["m_obs"].to_numpy()[test_m] > 0.5
    ev_te = ev[test_m]
    ite = torch.tensor(np.flatnonzero(test_m), device=dev)
    rows, hist_rows = [], []
    # 최종 학습을 시드마다 반복 — 조합끼리 **같은 시드로 짝비교**하려고. 첫 시드 모델만 XAI 용으로 저장
    for si, seed in enumerate(seeds):
        tune.set_seed(seed)
        hist: list = []
        net, best_loss = T.fit(train, D, var_t61, tr_all, va_all, hp, epochs=fe, patience=fp,
                               n_mc=config.N_MC, dev=dev, log_every=50 if si == 0 else None, history=hist,
                               arch=arch, dip=dip)
        hist_rows += [{"시드": seed, **h} for h in hist]
        p_va = T.predict_prob(net, D, va_all, dev)
        p_te_raw = T.predict_prob(net, D, test_m, dev)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_va, ev[va_all])
        p_te = np.clip(iso.predict(p_te_raw), 0, 1)
        mu_te, sd_te = T.predict_mean_sd(net, D, ite)
        out = {"계열": train, "정제": refine_name, "시드": seed, "해상도_분": config.RES_MIN,
               "구조": arch, "타깃": target,
               "CV_AUC": round(study.best_value, 4),
               "완료trial": int((trials["state"] == "COMPLETE").sum()),
               "가지치기trial": int((trials["state"] == "PRUNED").sum()),
               "시험시작": str(t_cut.date()), "시험n": int(test_m.sum()), "시험사건": int(ev_te.sum()),
               "μ_JT": round(net.mu_jt.item(), 4), "UA_배관_kW/K": round(net.ua_kw.item(), 2),
               "N₀": round(net.n0.item(), 3), "T_g": f"{net.tg_a.item():+.2f}{net.tg_b.item():+.3f}·T_in",
               "공급온도예측RMSE": round(float(np.sqrt(np.mean((mu_te - y[test_m]) ** 2))), 2),
               **{f"hp_{k}": v for k, v in hp.items()}}
        for tag, mask in (("전체", np.ones_like(obs_te)), ("유량관측", obs_te), ("유량미관측", ~obs_te)):
            for name, p in (("PINN", p_te), ("GBM", g_te)):
                for k, v in T.metrics(p[mask], ev_te[mask], base).items():
                    out[f"{name}_{tag}_{k}"] = v
        if arch in ("nn", "lstm"):                        # 물리층이 없다 — 모수는 초기값 그대로라 비운다
            out.update({"μ_JT": np.nan, "UA_배관_kW/K": np.nan, "N₀": np.nan, "T_g": ""})
        rows.append(out)
        log.info("[%s|%s] 시드 %d  시험 AUC PINN %.3f / GBM %.3f · 리프트1 %.1f / %.1f · Brier개선 %.1f%%",
                 train, refine_name, seed, out["PINN_전체_AUC"], out["GBM_전체_AUC"],
                 out["PINN_전체_리프트1"], out["GBM_전체_리프트1"], out["PINN_전체_Brier개선%"])
        if si > 0:
            del net
            continue

        # --- XAI 용 산출 (첫 시드)
        idx_bg = np.flatnonzero(tr_all)
        rng = np.random.default_rng(config.RANDOM_SEED)
        bg = rng.choice(idx_bg, size=min(2000, len(idx_bg)), replace=False)
        torch.save({"state": net.state_dict(), "hp": hp, "refine": list(refine), "cols": cols,
                    "arch": arch, "dip": dip, "target": target,
                    "names": names, "n_feat": int(D["Z"].shape[-1]),
                    "ua_init_kw": config.UA_PIPE_INIT_KW[train]}, out_dir / f"model_{train}.pt")
        np.savez_compressed(out_dir / f"xai_{train}.npz",
                            Z_test=D["Z"][ite].cpu().numpy(), Z_bg=D["Z"][torch.tensor(bg, device=dev)].cpu().numpy(),
                            now_test=D["now"][ite].cpu().numpy(), ev_test=ev_te, p_test=p_te, p_gbm=g_te,
                            m_obs=obs_te, t_index=np.asarray(X.index[test_m].astype("int64")))
        pd.DataFrame({"p": p_te, "p_gbm": g_te, "ev": ev_te, "mu": mu_te, "sd": sd_te, "y": y[test_m],
                      "m_obs": obs_te},
                     index=X.index[test_m]).to_parquet(out_dir / f"pred_{train}.parquet")
        with open(out_dir / f"best_{train}.json", "w", encoding="utf-8") as fh:
            json.dump({"refine": refine_name, "hp": hp, "cv_auc": study.best_value, "seeds": list(seeds),
                       "arch": arch, "target": target,
                       "res_min": config.RES_MIN,
                       "folds": [{k: v for k, v in fd.items() if k in ("tag",)} | {"fit_end": str(fd["fit_end"])}
                                 for fd in fl]}, fh, ensure_ascii=False, indent=1)
        del net
    pd.DataFrame(hist_rows).to_csv(out_dir / f"history_{train}.csv", index=False, encoding="utf-8-sig")
    if dev == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trains", nargs="+", default=list(config.TRAINS))
    ap.add_argument("--prep-only", action="store_true", help="fold 경계별 KF·AE·EKF 산출만 하고 끝낸다")
    ap.add_argument("--variant", default="auto", choices=["auto", *tune.REFINE_CHOICES],
                    help="정제 조합 고정 (auto = Optuna 가 고름)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[config.RANDOM_SEED])
    ap.add_argument("--arch", default="hard", choices=config.ARCHS)
    ap.add_argument("--target", default="min_raw", choices=config.TARGETS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = config.variant_dir(args.variant, args.smoke, args.arch, args.target)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in args.trains:
        r = run_train(t, out_dir, args.smoke, dev, prep_only=args.prep_only, variant=args.variant,
                      seeds=args.seeds, arch=args.arch, target=args.target)
        if not args.prep_only:
            # 계열별 파일 — 같은 폴더에서 M·Z 를 병렬로 돌려도 서로 덮어쓰지 않는다
            pd.DataFrame(r).to_csv(out_dir / f"metrics_seeds_{t}.csv", index=False, encoding="utf-8-sig")
        rows += r
    if args.prep_only:
        log.info("준비 완료: %s", rows)
        return
    # metrics.csv = 폴더에 있는 계열별 결과의 첫 시드 행 (XAI 그림이 읽는 모델과 같은 시드)
    m = pd.concat([pd.read_csv(p).groupby("계열").head(1) for p in sorted(out_dir.glob("metrics_seeds_*.csv"))])
    m.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    m = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    show = ["계열", "정제", "CV_AUC", "완료trial", "가지치기trial", "시험시작", "시험n", "시험사건",
            "PINN_전체_AUC", "GBM_전체_AUC", "PINN_전체_리프트1", "GBM_전체_리프트1",
            "PINN_전체_리프트5", "PINN_전체_Brier개선%", "공급온도예측RMSE", "μ_JT", "UA_배관_kW/K", "N₀"]
    print("\n=== 최종 (시험 20%) ===")
    print(m[[c for c in show if c in m.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
