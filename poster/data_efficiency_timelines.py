"""2-year training case — freezing-probability timelines, KF–soft PINN vs gradient boosting / neural network (line M).

Same style as results_compare/*_b_probability_test_period.png. Predictions: seed-42 models retrained on the 2-year
window (`python -m pinn.restricted.data_efficiency --years 2 --archs gbm softw nn --seeds 42 --save-pred`).
Note lines: test AUC = 5-seed mean ± SD from the data-efficiency run; alarm hits = the plotted (seed-42) model,
checked against metrics.csv.

    PYTHONPATH=. .venv/bin/python poster/data_efficiency_timelines.py
      → ttttt/poster_graphs/results_data_efficiency/2y_KF-soft-PINN_vs_{Gradient-boosting,Neural-network}_probability.png
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poster.results_model_timelines import INK, INK2, LAST_VALID, ROOT, new_ax

RUN = ROOT / "pinn" / "restricted" / "output" / "optuna" / "data_efficiency"
OUT = ROOT / "ttttt" / "poster_graphs" / "results_data_efficiency"
YEARS, SEED = "2", 42
MODELS = {"softw": ("KF–soft PINN (2-year training)", "#2a78d6"),
          "nn": ("Neural network (2-year training)", "#c98500"),
          "gbm": ("Gradient boosting (2-year training)", "#eb6834")}


def load(model: str) -> dict:
    pred = pd.read_parquet(RUN / f"pred_{YEARS}_{model}_{SEED}.parquet").sort_index()
    ev = (pred["ev"] > 0.5).to_numpy()
    k = int(np.ceil(0.01 * len(pred)))
    tie = np.random.default_rng(SEED).random(len(pred)) if model == "gbm" else pred["mu"].to_numpy()
    top = np.zeros(len(pred), bool)
    top[np.lexsort((tie, -pred["p"].to_numpy()))[:k]] = True
    hits = int((top & ev).sum())
    m = pd.read_csv(RUN / "metrics.csv")
    m = m[(m["years"].astype(str) == YEARS) & (m["model"] == model)]
    assert hits == int(m.loc[m["seed"] == SEED, "alarm_hits"].iloc[0]), f"{model}: hits {hits} ≠ metrics.csv"
    auc = f"{m['AUC'].mean():.3f}" + (f" ± {m['AUC'].std(ddof=1):.3f}" if len(m) > 1 else "")
    label, color = MODELS[model]
    return {"pred": pred, "thr": float(np.sort(pred["p"].to_numpy())[::-1][k - 1]), "hits": hits,
            "events": int(ev.sum()), "auc": auc, "label": label, "color": color}


def main():
    B = load("softw")
    idx = pd.date_range(B["pred"].index.min(), LAST_VALID, freq="1h")
    for other, fname in (("gbm", "Gradient-boosting"), ("nn", "Neural-network")):
        C = load(other)
        fig, ax = new_ax(14)
        ax.plot(idx, C["pred"]["p"].reindex(idx), color=C["color"], lw=0.55, label=C["label"])
        ax.plot(idx, B["pred"]["p"].reindex(idx), color=B["color"], lw=0.55, label=B["label"])
        ax.axhline(C["thr"], color=C["color"], lw=1.1, ls=(0, (5, 4)), alpha=0.8)
        ax.axhline(B["thr"], color=B["color"], lw=1.1, ls=(0, (5, 4)), alpha=0.8)
        ax.plot([], [], color=INK2, lw=1.1, ls=(0, (5, 4)), label="Top-1% alarm threshold (each model)")
        evw = B["pred"]["ev"].reindex(idx).to_numpy() > 0.5
        ax.vlines(idx[evw], -0.09, -0.03, color=INK, lw=0.7, label="Actual freezing")
        ax.set_ylim(-0.11, 1.45)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Freezing probability\nwithin 6 h")
        for j, M_ in enumerate((B, C)):
            ax.text(0.005, 0.99 - j * 0.085, f"{M_['label']}: test AUC {M_['auc']}  ·  top-1% alarms caught "
                    f"{M_['hits']} of {M_['events']} h", transform=ax.transAxes, va="top", fontsize=10.5,
                    color=M_["color"])
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.22))
        fig.tight_layout()
        name = f"2y_KF-soft-PINN_vs_{fname}_probability.png"
        fig.savefig(OUT / name, dpi=300, facecolor="white")
        plt.close(fig)
        print("saved:", name, "| hits", B["hits"], "vs", C["hits"], "| AUC", B["auc"], "vs", C["auc"])


if __name__ == "__main__":
    main()
