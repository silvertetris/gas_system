"""Research Results — per-model test-period timelines, one panel per file (English, white background, 300 dpi).

Models (line M, target "min outlet gas temperature < 0 °C within 6 h", same test period and inputs per refinement):
  KF–soft PINN (final) · AE–soft PINN · KF–hard PINN · Neural network (KF, no physics) · Gradient boosting (KF inputs)
Per model: full-period temperature · full-period freezing probability · 10-day zoom temperature · 10-day zoom
probability (GBM has no temperature output → probability only). Same axes and zoom window for every model.
Alarm hits use the equal-budget rule of compare_timeline.py (top 1% of test hours; PINN ties by lower predicted
mean, GBM ties by seeded RNG). Hours after the last valid measurement (2026-04-20 13:00, logger frozen) are not drawn.

    PYTHONPATH=. .venv/bin/python poster/results_model_timelines.py   → ttttt/poster_graphs/results_models/
"""
from __future__ import annotations

import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT = ROOT / "pinn" / "restricted" / "output" / "optuna"
ACC = ROOT / "ttttt" / "5_모델비교" / "model_accuracy_min.csv"
OUT = ROOT / "ttttt" / "poster_graphs" / "results_models"
LAST_VALID = pd.Timestamp("2026-04-20 13:00")
ZOOM = (pd.Timestamp("2023-12-08 18:00"), pd.Timestamp("2023-12-18 18:00"))
SEED = 42

INK, INK2, MUTED, AXIS, GRID = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e4e3df"
ORANGE = "#eb6834"
# (file key, label, pred dir, prob column, colour, csv refinement, csv model name)
MODELS = [
    ("1_KF-soft-PINN", "KF–soft PINN (final)", "softw_min/v_KF", "p", "#2a78d6", "KF", "soft PINN (물리 가중 확장)"),
    ("2_AE-soft-PINN", "AE–soft PINN", "softw_min/v_AE", "p", "#1baf7a", "AE", "soft PINN (물리 가중 확장)"),
    ("3_KF-hard-PINN", "KF–hard PINN", "hard_min/v_KF", "p", "#4a3aa7", "KF", "hard PINN (출력 = 물리식)"),
    ("4_Neural-network", "Neural network (no physics)", "nn_min/v_KF", "p", "#c98500", "KF", "nn (물리 없음)"),
    ("5_Gradient-boosting", "Gradient boosting", "softw_min/v_KF", "p_gbm", ORANGE, "KF", "GBM (같은 입력)"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "legend.labelcolor": INK2})


def new_ax(w, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=300, facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)
    print("saved:", name)


def alarms(pred: pd.DataFrame, pcol: str, k: int) -> np.ndarray:
    top = np.zeros(len(pred), bool)
    if pcol == "p_gbm":
        tie = np.random.default_rng(SEED).random(len(pred))
    else:
        tie = pred["mu"].to_numpy()
    top[np.lexsort((tie, -pred[pcol].to_numpy()))[:k]] = True
    return top


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acc = pd.read_csv(ACC, encoding="utf-8-sig")
    acc = acc[acc["계열"] == "M"].set_index(["정제", "모델"])
    preds = {key: pd.read_parquet(OPT / d / "pred_M.parquet").sort_index() for key, _, d, *_ in MODELS}

    # shared axes
    temp_lo = min(-4.0, min(float(p["mu"].min()) for p in preds.values()) - 1)
    temp_hi = max(float(p.loc[:LAST_VALID, ["mu", "y"]].max().max()) for p in preds.values()) + 2
    zoom_frames = {k: p.loc[ZOOM[0]:ZOOM[1]] for k, p in preds.items()}
    zlo = min(float((z["mu"] - z["sd"]).min()) for z in zoom_frames.values()) - 0.5
    zhi = max(float((z["mu"] + z["sd"]).max()) for z in zoom_frames.values()) + 0.5

    rows = []
    for key, label, d, pcol, color, ref, csv_name in MODELS:
        pred = preds[key]
        ev = (pred["ev"] > 0.5).to_numpy()
        k = int(np.ceil(0.01 * len(pred)))
        top = alarms(pred, pcol, k)
        hits = int((top & ev).sum())
        thr = float(np.sort(pred[pcol].to_numpy())[::-1][k - 1])
        auc_txt = str(acc.loc[(ref, csv_name), "시험 AUC"]).replace("±", "±")
        table_hits = int(acc.loc[(ref, csv_name), "경보가 잡은 빙결"])
        assert hits == table_hits, f"{label}: alarm hits {hits} ≠ table {table_hits}"
        rows.append((label, auc_txt, round(roc_auc_score(ev, pred[pcol]), 3), hits, int(ev.sum())))
        note = f"Test AUC {auc_txt}  ·  Top-1% alarms caught {hits} of {int(ev.sum())} freezing hours"

        full = pd.date_range(pred.index.min(), LAST_VALID, freq="1h")
        w = pred.reindex(full)
        wz = pred.reindex(pd.date_range(ZOOM[0], ZOOM[1], freq="1h"))
        has_temp = pcol == "p"

        if has_temp:  # (a) full-period temperature
            fig, ax = new_ax(14)
            ax.plot(w.index, w["y"], color=INK2, lw=0.5, label="Actual")
            ax.plot(w.index, w["mu"], color=color, lw=0.6, label=f"Predicted — {label}")
            e = w[w["ev"] > 0.5]
            ax.scatter(e.index, e["y"], s=8, color=ORANGE if color != ORANGE else INK, lw=0, zorder=4,
                       label="Freezing (< 0 °C)")
            ax.axhline(0, color=AXIS, lw=1)
            ax.set_ylim(temp_lo, temp_hi)
            ax.set_ylabel("Min. outlet gas temp.\nin next 6 h [°C]")
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.22), markerscale=2)
            save(fig, f"{key}_a_temperature_test_period.png")

        # (b) full-period probability
        fig, ax = new_ax(14)
        ax.plot(w.index, w[pcol], color=color, lw=0.7, label=f"Freezing probability — {label}")
        ax.axhline(thr, color=MUTED, lw=1.2, ls=(0, (5, 4)), label="Top-1% alarm threshold")
        ax.vlines(w.index[w["ev"] > 0.5], -0.09, -0.03, color=INK, lw=0.7, label="Actual freezing")
        ax.set_ylim(-0.11, 1.3)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Freezing probability\nwithin 6 h")
        ax.text(0.005, 0.99, note, transform=ax.transAxes, va="top", fontsize=11, color=INK2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.22))
        save(fig, f"{key}_b_probability_test_period.png")

        if has_temp:  # (c) zoom temperature
            fig, ax = new_ax(12)
            ax.fill_between(wz.index, wz["mu"] - wz["sd"], wz["mu"] + wz["sd"], color=color, alpha=0.15, lw=0,
                            label="Predicted ±1σ")
            ax.plot(wz.index, wz["y"], color=INK2, lw=0.9, label="Actual")
            ax.plot(wz.index, wz["mu"], color=color, lw=1.3, label=f"Predicted — {label}")
            e = wz[wz["ev"] > 0.5]
            ax.scatter(e.index, e["y"], s=18, color=ORANGE, lw=0, zorder=4, label="Freezing (< 0 °C)")
            ax.axhline(0, color=AXIS, lw=1)
            ax.set_ylim(zlo, zhi)
            ax.set_ylabel("Min. outlet gas temp.\nin next 6 h [°C]")
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.22))
            save(fig, f"{key}_c_temperature_10days.png")

        # (d) zoom probability
        fig, ax = new_ax(12)
        ax.plot(wz.index, wz[pcol], color=color, lw=1.3, label=f"Freezing probability — {label}")
        ax.axhline(thr, color=MUTED, lw=1.2, ls=(0, (5, 4)), label="Top-1% alarm threshold")
        ax.vlines(wz.index[wz["ev"] > 0.5], -0.09, -0.03, color=INK, lw=0.9, label="Actual freezing")
        ax.set_ylim(-0.11, 1.05)
        ax.set_ylabel("Freezing probability\nwithin 6 h")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.22))
        save(fig, f"{key}_d_probability_10days.png")

    print(pd.DataFrame(rows, columns=["model", "test AUC (5 seeds)", "AUC seed 42", "alarm hits", "freezing h"]))


if __name__ == "__main__":
    main()
