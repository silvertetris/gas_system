"""빙결 타깃 해부 + 물리식 상한 시험 (2026-09-13). 미래 상태를 완벽히 안다고 가정했을 때 물리식 T61 의 사건 판별력."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pinn.restricted import config, data, ekf
h = config.rows(config.HORIZON_H)
for t in ("M", "Z"):
    f = data.build_base(t).copy()
    f = f[f.index >= pd.Timestamp(config.VALID_FROM)]
    def windows(a):
        x = np.concatenate([a[1:].astype(np.float64), np.full(h, np.nan)])
        return np.lib.stride_tricks.sliding_window_view(x, h)[:len(a)]
    tw = windows(f["t61_min"].to_numpy()); valid = np.isfinite(tw).sum(1) >= max(h // 2, 1)
    arg = np.where(valid, np.nanargmin(np.where(np.isfinite(tw), tw, np.inf), axis=1), 0); rows = np.arange(len(f))
    at = lambda c: np.where(valid, windows(f[c].to_numpy())[rows, arg], np.nan)
    f["y_t61_mean"] = at("t61"); f["y_p61"] = at("p61"); f["y_mobs"] = at("m_obs")
    ok = f[["t61", "t61_min", "hdr", "t_in", "dp", "y_t61_min", "y_hdr", "y_dp", "y_tin"]].notna().all(axis=1)
    g = f[ok]; ev = (g["y_t61_min"] < 0).to_numpy()
    E = g[ev]
    th = pd.read_csv(f"pinn/restricted/output/optuna/folds/ekf_params_{t}_final.csv").iloc[0]
    theta = [th["μ_JT"], th["UA_W_per_K"], th["N0"], th["a"], th["b"]]
    print(f"\n===== {t}  행 {len(g)}  사건 {ev.sum()} ({ev.mean():.2%}) | 물리모수 μ={theta[0]:.3f} UA={theta[1]/1e3:.2f}kW/K N0={theta[2]:.3f} (exp(-N0)={np.exp(-theta[2]):.2f}) a={theta[3]:.2f} b={theta[4]:.3f}")
    print("[사건 해부]")
    print(f"  사건 시각의 1시간 평균 T61 ≥ 0 (분 단위 순간 하강만): {np.mean(E['y_t61_mean'] >= 0):.1%}")
    print(f"  1시간 평균 T61 < 0 (시간 평균까지 영하): {np.mean(E['y_t61_mean'] < 0):.1%}")
    print(f"  최저값 분위(5/25/50/75/95%): {np.round(np.nanquantile(E['y_t61_min'], [.05,.25,.5,.75,.95]),1).tolist()}")
    print(f"  최저값 < -10℃: {np.mean(E['y_t61_min'] < -10):.1%}  | 순간값-시간평균 차 중앙 {np.nanmedian(E['y_t61_mean']-E['y_t61_min']):.1f}℃")
    print(f"  사건 시각 헤더온도 <5℃: {np.mean(E['y_hdr'] < 5):.1%}  <10℃: {np.mean(E['y_hdr'] < 10):.1%}  중앙 {np.nanmedian(E['y_hdr']):.1f}℃")
    print(f"  사건 시각 유량 관측: {np.mean(E['y_mobs'] > .5):.1%}  | p61<=0: {np.mean(E['y_p61'] <= 0):.1%}")
    # 연속 시간 (현재 시각 t61_min<0 기준)
    neg = (f["t61_min"] < 0).astype(int).to_numpy()
    runs = []; c = 0
    for v in neg:
        if v: c += 1
        elif c: runs.append(c); c = 0
    runs = np.array(runs)
    print(f"  영하 구간 {len(runs)}개: 1시간짜리 {np.mean(runs==1):.1%}, 2~3시간 {np.mean((runs>=2)&(runs<=3)):.1%}, 6시간+ {np.mean(runs>=6):.1%}")
    # 연도별 사건
    print("  연도별 사건률:", {y: f"{v:.2%}" for y, v in pd.Series(ev, index=g.index).groupby(g.index.year).mean().items()})
    print("[상한 시험 — 미래 상태를 완벽히 안다고 가정]")
    obs = g["y_log_m"].notna().to_numpy()
    tp = ekf.physics(theta, g["y_hdr"].to_numpy(), g["y_dp"].to_numpy(), g["y_tin"].to_numpy(),
                     np.where(obs, g["y_log_m"].to_numpy(), np.nanmedian(g["log_m"])))[0]
    def auc(score, m=None):
        m = np.ones(len(g), bool) if m is None else m
        s = score[m]; e = ev[m]; ok_ = np.isfinite(s)
        return roc_auc_score(e[ok_], s[ok_])
    test = np.asarray(g.index >= pd.Timestamp("2023-08-03"))
    for name, sc in (("물리식 T61(완벽한 미래 상태)", -tp), ("미래 헤더온도만", -g["y_hdr"].to_numpy()),
                     ("미래 1시간평균 T61 (완벽 예보)", -g["y_t61_mean"].to_numpy()),
                     ("현재 t61_min (지속 예보)", -g["t61_min"].to_numpy()), ("현재 헤더온도", -g["hdr"].to_numpy())):
        print(f"  {name:32s} AUC 전체 {auc(sc):.3f} | 유량관측 {auc(sc, obs):.3f} | 시험 {auc(sc, test):.3f}")
    r = tp[obs] - g["y_t61_mean"].to_numpy()[obs]
    print(f"  물리식 vs 실제 1시간평균 T61 잔차: RMSE {np.sqrt(np.nanmean(r**2)):.2f}℃, 사건행 편향 {np.nanmean((tp - g['y_t61_min'].to_numpy())[obs & ev]):+.2f}℃ (물리식 − 순간최저)")
