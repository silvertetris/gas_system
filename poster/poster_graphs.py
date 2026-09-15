"""Poster graphs — English, white background, 300 dpi, one chart per file → ttttt/poster_graphs/.

Final model = weighted soft PINN (w ≤ 30) + KF refinement, line M, target "min supply temperature < 0 °C within 6 h".
Sources: pinn/restricted/output/optuna/softw_min/v_*/ (pred · history · trials · fold_params · physics_decomp · shap)
and ttttt/5_모델비교/model_accuracy_min.csv (all architectures, same inputs). Alarm hits use the same equal-budget
rule as compare_timeline.py (top 1% of test hours; PINN ties broken by lower predicted mean, GBM ties by seeded RNG).

    PYTHONPATH=. .venv/bin/python poster/poster_graphs.py
"""
from __future__ import annotations

import pathlib
import shutil
import unicodedata

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT = ROOT / "pinn" / "restricted" / "output" / "optuna" / "softw_min"
FINAL = OPT / "v_KF"
ACC = ROOT / "ttttt" / "5_모델비교" / "model_accuracy_min.csv"
OUT = ROOT / "ttttt" / "poster_graphs"
TRAIN = "M"
SEED = 42
Y_FLOOR = -12.0

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#c3c2b7"
BLUE, ORANGE, AQUA, GRAY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#c9c8c2"

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 12,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "legend.frameon": False, "legend.labelcolor": INK2})


def fig_ax(w=10, h=4.5, **kw):
    fig, ax = plt.subplots(figsize=(w, h), dpi=300, facecolor="white", **kw)
    for a in np.atleast_1d(ax):
        style(a)
    return fig, ax


def style(ax, grid="y"):
    ax.set_facecolor("white")
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)
    print("saved:", name)


def nfc_dir(base: pathlib.Path, name: str) -> pathlib.Path:
    want = unicodedata.normalize("NFC", name)
    return next(p for p in base.iterdir() if unicodedata.normalize("NFC", p.name) == want)


# ---------------------------------------------------------------- data
def load_pred():
    pred = pd.read_parquet(FINAL / f"pred_{TRAIN}.parquet").sort_index()
    ev = (pred["ev"] > 0.5).to_numpy()
    k = int(np.ceil(0.01 * len(pred)))
    rng = np.random.default_rng(SEED)
    top_p = np.zeros(len(pred), bool)
    top_p[np.lexsort((pred["mu"].to_numpy(), -pred["p"].to_numpy()))[:k]] = True
    top_g = np.zeros(len(pred), bool)
    top_g[np.lexsort((rng.random(len(pred)), -pred["p_gbm"].to_numpy()))[:k]] = True
    pred = pred.assign(alarm=top_p, alarm_gbm=top_g, evb=ev)
    print(f"test hours {len(pred)} · freezing {ev.sum()} · alarms {k} · caught PINN {int((top_p & ev).sum())} "
          f"/ GBM {int((top_g & ev).sum())}")
    return pred, k


MODEL_ROWS = [  # (label in csv, poster label, colour)
    ("GBM (같은 입력)", "Gradient boosting", ORANGE),
    ("hard PINN (출력 = 물리식)", "Hard PINN", GRAY),
    ("soft PINN (물리 = 손실)", "Soft PINN (w ≤ 3)", GRAY),
    ("soft PINN (물리 가중 확장)", "KF–soft PINN (final)", BLUE),
    ("nn (물리 없음)", "Neural network (no physics)", GRAY),
    ("LSTM (시계열, 물리 없음)", "LSTM (no physics)", GRAY),
]


def load_models():
    a = pd.read_csv(ACC, encoding="utf-8-sig")
    a = a[(a["계열"] == TRAIN) & (a["정제"] == "KF")].set_index("모델")
    rows = []
    for key, label, c in MODEL_ROWS:
        r = a.loc[key]
        auc = str(r["시험 AUC"]).split("±")
        rows.append({"label": label, "color": c, "auc": float(auc[0]),
                     "sd": float(auc[1]) if len(auc) > 1 else np.nan,
                     "hits": int(r["경보가 잡은 빙결"]), "events": int(r["빙결 시간"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experimental details
def mu_jt_convergence():
    h = pd.read_csv(FINAL / f"history_{TRAIN}.csv", encoding="utf-8-sig").rename(columns={"시드": "seed"})
    fig, ax = fig_ax(10, 4.2)
    for s, g in h.groupby("seed"):
        main = s == SEED
        ax.plot(g["epoch"], g["μ_JT"], color=BLUE, lw=2.2 if main else 1.2, alpha=1 if main else 0.35,
                label="Learned μ_JT (seed 42)" if main else None)
    ax.plot([], [], color=BLUE, lw=1.2, alpha=0.35, label="Learned μ_JT (other seeds)")
    ax.axhline(0.56, color=ORANGE, lw=1.8, ls=(0, (5, 4)), label="Literature value 0.56 °C/bar")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Joule–Thomson coefficient\nμ_JT [°C/bar]")
    ax.legend(loc="lower right")
    save(fig, "03_mu_JT_convergence.png")


def optuna_history():
    t = pd.read_csv(FINAL / f"trials_{TRAIN}.csv", encoding="utf-8-sig")
    done = t[t["state"] == "COMPLETE"]
    fig, ax = fig_ax(10, 4.2)
    ax.scatter(done["number"], done["value"], s=46, color=BLUE, zorder=3, edgecolor="white", lw=1, label="Trial")
    ax.step(done["number"], done["value"].cummax(), where="post", color=ORANGE, lw=2.2, label="Best so far")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Cross-validation AUC\n(5-fold mean)")
    ax.legend(loc="lower right")
    save(fig, "04_optuna_history.png")


def fold_auc():
    fp = pd.read_csv(FINAL / f"fold_params_{TRAIN}.csv", encoding="utf-8-sig")
    k = fp["fold"].str.replace("fold", "").astype(int)
    fig, ax = fig_ax(8, 4.2)
    ax.bar(k, fp["검증AUC"], color=BLUE, width=0.6)
    for xi, v, e in zip(k, fp["검증AUC"], fp["검증사건"]):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=11, color=INK)
        ax.text(xi, 0.04, f"{e} h", ha="center", va="bottom", fontsize=10, color="white")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(k, [f"Fold {i}" for i in k])
    ax.set_ylabel("Validation AUC")
    ax.text(0.99, -0.16, "Number in bar = freezing hours in validation window", transform=ax.transAxes, ha="right",
            fontsize=10, color=MUTED)
    save(fig, "05_fold_validation_auc.png")


def refinement_comparison():
    names = [("없음", "None"), ("KF", "KF"), ("AE", "AE"), ("EKF", "EKF"), ("KF+AE+EKF", "KF+AE+EKF")]
    rows = []
    for key, label in names:
        m = pd.read_csv(nfc_dir(OPT, f"v_{key}") / f"metrics_seeds_{TRAIN}.csv", encoding="utf-8-sig")
        rows.append((label, m["CV_AUC"].iloc[0], m["PINN_전체_AUC"].mean(), m["PINN_전체_AUC"].std(ddof=1)))
    d = pd.DataFrame(rows, columns=["label", "cv", "test", "sd"])
    fig, axes = fig_ax(11, 4.2, ncols=2, sharey=True)
    y = np.arange(len(d))[::-1]
    cols = [BLUE if l == "KF" else GRAY for l in d["label"]]
    ax = axes[0]
    ax.scatter(d["cv"], y, s=110, color=cols, zorder=3, edgecolor="white", lw=1.5)
    for yi, v in zip(y, d["cv"]):
        ax.text(v + 0.0006, yi + 0.22, f"{v:.3f}", fontsize=10.5, color=INK2, ha="center")
    ax.set_xlim(0.935, 0.95)
    ax.set_xlabel("Cross-validation AUC\n(selection criterion)")
    ax.set_yticks(y, d["label"])
    ax = axes[1]
    ax.errorbar(d["test"], y, xerr=d["sd"], fmt="none", ecolor=LIGHT, elinewidth=2, capsize=4)
    ax.scatter(d["test"], y, s=110, color=cols, zorder=3, edgecolor="white", lw=1.5)
    for yi, v in zip(y, d["test"]):
        ax.text(v, yi + 0.22, f"{v:.3f}", fontsize=10.5, color=INK2, ha="center")
    ax.set_xlim(0.89, 0.955)
    ax.set_xlabel("Test AUC\n(mean ± SD, 5 seeds)")
    for a in axes:
        a.grid(True, axis="x", color=GRID, lw=0.8); a.grid(False, axis="y")
    save(fig, "06_refinement_comparison.png")


# ---------------------------------------------------------------- results
def model_auc(m):
    d = m.iloc[::-1].reset_index(drop=True)
    fig, ax = fig_ax(10, 4.4)
    y = np.arange(len(d))
    ax.errorbar(d["auc"], y, xerr=d["sd"].fillna(0), fmt="none", ecolor=LIGHT, elinewidth=2.2, capsize=4)
    ax.scatter(d["auc"], y, s=130, color=d["color"], zorder=3, edgecolor="white", lw=1.5)
    for yi, v, s in zip(y, d["auc"], d["sd"]):
        ax.text(v + (s if np.isfinite(s) else 0) + 0.002, yi, f"{v:.3f}", va="center", fontsize=11, color=INK)
    ax.set_yticks(y, d["label"])
    ax.set_xlim(0.86, 0.96)
    ax.set_xlabel("Test AUC (mean ± SD over 5 seeds)")
    ax.grid(True, axis="x", color=GRID, lw=0.8); ax.grid(False, axis="y")
    save(fig, "07_model_comparison_AUC.png")


def model_hits(m):
    d = m.iloc[::-1].reset_index(drop=True)
    ev = int(d["events"].iloc[0])
    fig, ax = fig_ax(10, 4.4)
    y = np.arange(len(d))
    ax.barh(y, d["hits"], color=d["color"], height=0.6)
    for yi, v in zip(y, d["hits"]):
        ax.text(v + 1, yi, f"{v}", va="center", fontsize=11, color=INK)
    ax.set_yticks(y, d["label"])
    ax.set_xlim(0, max(d["hits"]) * 1.15)
    ax.set_xlabel(f"Freezing hours caught by top-1% alarms (of {ev} h)")
    ax.grid(True, axis="x", color=GRID, lw=0.8); ax.grid(False, axis="y")
    save(fig, "08_model_comparison_alarm_hits.png")


def _temp_panel(ax, w, band=False):
    if band:
        ax.fill_between(w.index, w["mu"] - w["sd"], w["mu"] + w["sd"], color=BLUE, alpha=0.15, lw=0,
                        label="Predicted ±1σ")
    ax.plot(w.index, w["y"].clip(lower=Y_FLOOR), color=INK2, lw=0.8 if band else 0.5, label="Actual")
    ax.plot(w.index, w["mu"], color=BLUE, lw=1.2 if band else 0.6, label="Predicted (KF–soft PINN)")
    e = w[w["ev"] > 0.5]
    ax.scatter(e.index, e["y"].clip(lower=Y_FLOOR), s=14 if band else 6, color=ORANGE, zorder=4, lw=0,
               label="Freezing (< 0 °C)")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("Min. supply gas\ntemperature in 6 h [°C]")


def _prob_panel(ax, w, thr):
    ax.plot(w.index, w["p_gbm"], color=ORANGE, lw=0.8, label="Gradient boosting")
    ax.plot(w.index, w["p"], color=BLUE, lw=0.9, label="KF–soft PINN")
    ax.axhline(thr, color=MUTED, lw=1.2, ls=(0, (5, 4)), label="Top-1% alarm threshold")
    e = w.index[w["ev"] > 0.5]
    ax.vlines(e, -0.08, -0.02, color=INK, lw=0.7, label="Actual freezing")
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("Freezing probability\nwithin 6 h")


def timeline(pred, k):
    full = pd.date_range(pred.index.min(), pred.index.max(), freq="1h")
    w = pred.reindex(full)
    thr = float(np.sort(pred["p"].to_numpy())[::-1][k - 1])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6.8), dpi=300, sharex=True, facecolor="white",
                                 gridspec_kw={"height_ratios": [1.1, 1]})
    style(a1); style(a2)
    _temp_panel(a1, w)
    a1.set_ylim(min(-4.0, float(np.nanmin(w["y"].clip(lower=Y_FLOOR))) - 1), float(np.nanquantile(w["y"], 0.999)) + 3)
    a1.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.22), markerscale=2)
    _prob_panel(a2, w, thr)
    a2.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.2))
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    save(fig, "09_test_period_timeline.png")

    roll = pd.Series(pred["evb"].astype(float)).reindex(full, fill_value=0).rolling("10D").sum()
    end = roll.idxmax()
    z = w.loc[end - pd.Timedelta(days=10): end]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.8), dpi=300, sharex=True, facecolor="white",
                                 gridspec_kw={"height_ratios": [1.1, 1]})
    style(a1); style(a2)
    _temp_panel(a1, z, band=True)
    a1.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.22))
    _prob_panel(a2, z, thr)
    a2.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.2))
    a2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    print(f"zoom window {z.index[0]} → {z.index[-1]}")
    save(fig, "10_test_zoom_10days.png")


def monthly_hits(pred):
    mon = pd.DataFrame({"ev": pred["evb"], "p": pred["alarm"] & pred["evb"], "g": pred["alarm_gbm"] & pred["evb"]},
                       index=pred.index).resample("MS").sum()
    mon = mon[mon["ev"] > 0]
    x = np.arange(len(mon))
    fig, ax = fig_ax(11, 4.4)
    ax.bar(x - 0.27, mon["ev"], width=0.26, color=LIGHT, label="Freezing hours")
    ax.bar(x, mon["p"], width=0.26, color=BLUE, label="Caught by KF–soft PINN")
    ax.bar(x + 0.27, mon["g"], width=0.26, color=ORANGE, label="Caught by gradient boosting")
    ax.set_xticks(x, [f"{d:%Y-%m}" for d in mon.index], rotation=45, ha="right")
    ax.set_ylabel("Hours")
    ax.legend(loc="upper right")
    save(fig, "11_monthly_alarm_hits.png")


def roc_pr(pred):
    ev = pred["evb"].astype(int).to_numpy()
    fig, (a1, a2) = fig_ax(11, 4.8, ncols=2)
    for col, c, name in (("p", BLUE, "KF–soft PINN"), ("p_gbm", ORANGE, "Gradient boosting")):
        fpr, tpr, _ = roc_curve(ev, pred[col])
        a1.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(ev, pred[col]):.3f})")
        pr, rc, _ = precision_recall_curve(ev, pred[col])
        a2.plot(rc, pr, color=c, lw=2, label=f"{name} (AP {average_precision_score(ev, pred[col]):.3f})")
    a1.plot([0, 1], [0, 1], color=AXIS, lw=1)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
    a1.legend(loc="lower right", fontsize=10.5)
    a2.axhline(ev.mean(), color=AXIS, lw=1)
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a2.legend(loc="upper right", fontsize=10.5)
    for a in (a1, a2):
        a.grid(True, axis="both", color=GRID, lw=0.8)
    save(fig, "12_ROC_PR_curves.png")


def physics_attribution():
    d = pd.read_csv(FINAL / f"physics_decomp_{TRAIN}.csv", encoding="utf-8-sig").set_index("사건")
    r = d.loc["사건−평시"]
    items = [("Ground heat exchange", r["지중"]), ("Joule–Thomson cooling", r["줄톰슨"]),
             ("Header temperature", r["헤더"]), ("Supply temperature (total)", r["T61_확실성등가"])]
    labels = [i[0] for i in items][::-1]
    vals = np.array([i[1] for i in items])[::-1]
    cols = [INK2 if l.startswith("Supply") else (BLUE if v < 0 else ORANGE) for l, v in zip(labels, vals)]
    fig, ax = fig_ax(9, 3.8)
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + (0.12 if v >= 0 else -0.12), yi, f"{v:+.2f} °C".replace("-", "\u2212"), va="center", ha="left" if v >= 0 else "right",
                fontsize=11, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(min(vals) * 1.35, max(max(vals) * 1.8, 1.5))
    ax.set_xlabel("Difference: freezing hours − normal hours [°C]")
    ax.grid(True, axis="x", color=GRID, lw=0.8); ax.grid(False, axis="y")
    save(fig, "13_physics_attribution.png")


GROUPS = {"현재계측": "Current measurements", "이력": "Recent history", "KF": "Kalman filter signals",
          "유량": "Gas flow", "계절·시각": "Season & time of day", "히터": "Heater operation"}
FEATURES = {"t61_min": "Supply temperature (hourly minimum)", "t61": "Supply temperature",
            "kf_t61": "KF-smoothed supply temperature", "m_obs": "Flow-observed flag",
            "결측_log_m": "Flow-missing flag", "hour_sin": "Time of day (sin)",
            "innov_t61": "KF innovation (supply temperature)", "t61_minr3": "Supply temperature, 3-h minimum",
            "t61_mean24": "Supply temperature, 24-h mean", "t61_minr24": "Supply temperature, 24-h minimum"}


def input_importance():
    g = pd.read_csv(FINAL / f"shap_groups_{TRAIN}.csv", encoding="utf-8-sig").sort_values("전체_비중%")
    fig, ax = fig_ax(9, 3.8)
    y = np.arange(len(g))
    ax.barh(y, g["전체_비중%"], color=[AQUA if n == "KF" else BLUE for n in g["그룹"]], height=0.6)
    for yi, v in zip(y, g["전체_비중%"]):
        ax.text(v + 0.4, yi, f"{v:.1f}%", va="center", fontsize=11, color=INK)
    ax.set_yticks(y, [GROUPS[n] for n in g["그룹"]])
    ax.set_xlim(0, g["전체_비중%"].max() * 1.18)
    ax.set_xlabel("Share of mean |SHAP| [%]")
    ax.grid(True, axis="x", color=GRID, lw=0.8); ax.grid(False, axis="y")
    save(fig, "14_input_group_importance.png")

    f = pd.read_csv(FINAL / f"shap_features_{TRAIN}.csv", encoding="utf-8-sig").head(10).iloc[::-1]
    fig, ax = fig_ax(9, 4.6)
    y = np.arange(len(f))
    ax.barh(y, f["전체|SHAP|"], color=[AQUA if n == "KF" else BLUE for n in f["그룹"]], height=0.6)
    ax.set_yticks(y, [FEATURES[n] for n in f["특징"]])
    ax.set_xlabel("Mean |SHAP| value")
    ax.plot([], [], color=AQUA, lw=8, label="Kalman filter signal")
    ax.plot([], [], color=BLUE, lw=8, label="Other input")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", color=GRID, lw=0.8); ax.grid(False, axis="y")
    save(fig, "15_top10_input_features.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in (("KF_filtering_M.png", "01_KF_filtering_result.png"),
                     ("PINN_loss_curve_M.png", "02_PINN_loss_curve.png")):
        shutil.copy2(ROOT / "ttttt" / src, OUT / dst)
        print("copied:", dst)
    mu_jt_convergence()
    optuna_history()
    fold_auc()
    refinement_comparison()
    m = load_models()
    print(m.to_string())
    model_auc(m)
    model_hits(m)
    pred, k = load_pred()
    timeline(pred, k)
    monthly_hits(pred)
    roc_pr(pred)
    physics_attribution()
    input_importance()


if __name__ == "__main__":
    main()
