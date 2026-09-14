"""바이패스 혼합 가설 검정 (2026-09-13). 사건 시각 β·혼합식 판별력 — 기각됨."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pinn.restricted import config, data
h = 6
for t in ("M", "Z"):
    f = data.build_base(t).copy(); f = f[f.index >= pd.Timestamp(config.VALID_FROM)]
    def windows(a):
        x = np.concatenate([a[1:].astype(np.float64), np.full(h, np.nan)])
        return np.lib.stride_tricks.sliding_window_view(x, h)[:len(a)]
    tw = windows(f["t61_min"].to_numpy()); valid = np.isfinite(tw).sum(1) >= 3
    arg = np.where(valid, np.nanargmin(np.where(np.isfinite(tw), tw, np.inf), axis=1), 0); rows = np.arange(len(f))
    at = lambda c: np.where(valid, windows(f[c].to_numpy())[rows, arg], np.nan)
    for c in ("beta", "valve", "t61"): f[f"y_{c}"] = at(c)
    duty = [c for c in f.columns if c.startswith("duty_")]
    f["duty_mean"] = f[duty].mean(axis=1); f["y_duty"] = at("duty_mean")
    ok = f[["t61", "t61_min", "hdr", "t_in", "dp", "y_t61_min", "y_hdr", "y_dp", "y_tin"]].notna().all(axis=1)
    g = f[ok]; ev = (g["y_t61_min"] < 0).to_numpy(); E, N = g[ev], g[~ev]
    test = np.asarray(g.index >= pd.Timestamp("2023-08-03"))
    print(f"\n===== {t}")
    for c in ("y_beta", "y_valve", "y_duty", "y_hdr", "y_tin", "y_dp"):
        print(f"  {c:8s} 사건 중앙 {np.nanmedian(E[c]):8.3f} | 평시 중앙 {np.nanmedian(N[c]):8.3f} | 사건 결측 {E[c].isna().mean():.1%}")
    b = g["y_beta"].clip(0, 1).to_numpy()
    for mu in (0.45, 0.56):
        tmix = (1 - b) * g["y_hdr"].to_numpy() + b * (g["y_tin"].to_numpy() - mu * g["y_dp"].to_numpy())
        tmix_jt = tmix - (1 - b) * 0   # 헤더 쪽 JT 는 헤더 이후 감압이 없다고 가정
        m = np.isfinite(tmix)
        print(f"  혼합식 T=(1-β)·T_hdr+β·(T_in-{mu}·ΔP)  AUC 전체 {roc_auc_score(ev[m], -tmix[m]):.3f} | 시험 {roc_auc_score(ev[m & test], -tmix[m & test]):.3f} | "
              f"사건 중앙 {np.nanmedian(tmix[ev]):.1f}℃ / 평시 {np.nanmedian(tmix[~ev]):.1f}℃ | vs 실제 1h평균 RMSE {np.sqrt(np.nanmean((tmix - g['y_t61'].to_numpy())**2)):.2f}℃")
    m = np.isfinite(b)
    print(f"  미래 β 단독 AUC 전체 {roc_auc_score(ev[m], b[m]):.3f} | 시험 {roc_auc_score(ev[m & test], b[m & test]):.3f}")
    qs = pd.qcut(pd.Series(b[m]), 5, duplicates="drop")
    print("  β 5분위별 사건률:", {str(k): f"{v:.2%}" for k, v in pd.Series(ev[m]).groupby(qs.values, observed=True).mean().items()})
