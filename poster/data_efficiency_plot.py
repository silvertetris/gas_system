"""Data-efficiency result — KF–soft PINN vs neural network vs gradient boosting with less training data (line M).

Reads pinn/restricted/output/optuna/data_efficiency/metrics.csv (see pinn/restricted/data_efficiency.py):
same test period and same early-stopping/calibration block for every window; only the training window shrinks.
One chart per file, English, white background, 300 dpi → ttttt/poster_graphs/results_data_efficiency/
  a_test_AUC.png · b_alarm_hits.png · c_temperature_RMSE.png  (+ summary.csv)

    PYTHONPATH=. .venv/bin/python poster/data_efficiency_plot.py
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "pinn" / "restricted" / "output" / "optuna" / "data_efficiency" / "metrics.csv"
OUT = ROOT / "ttttt" / "poster_graphs" / "results_data_efficiency"
INK2, MUTED, AXIS, GRID = "#52514e", "#898781", "#c3c2b7", "#e4e3df"
MODELS = [("softw", "KF–soft PINN (final)", "#2a78d6", "o"), ("nn", "Neural network (no physics)", "#c98500", "s"),
          ("gbm", "Gradient boosting", "#eb6834", "D")]
ORDER = ["1", "2", "3", "5", "all"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "legend.labelcolor": INK2})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SRC)
    d["years"] = d["years"].astype(str)
    s = d.groupby(["years", "model"]).agg(AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std"),
                                           hits_mean=("alarm_hits", "mean"), hits_sd=("alarm_hits", "std"),
                                           rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), n=("seed", "size"),
                                           train_events=("train_events", "first"), span=("train_span", "first"))
    s.round(3).to_csv(OUT / "summary.csv")
    print(s.round(3).to_string())

    ev = d.groupby("years")["train_events"].first()
    span = d.groupby("years")["train_span"].first()
    order = [y for y in ORDER if y in ev.index]

    def tick(y):
        if y == "all":
            a, b = (pd.Timestamp(t) for t in span[y].split("~"))
            y_txt = f"All ({(b - a).days / 365.25:.1f} y)"
        else:
            y_txt = f"{y} y"
        return f"{y_txt}\n{int(ev[y])} freezing h"

    x = np.arange(len(order))
    for metric, ylabel, fname, skip in (("AUC", "Test AUC", "a_test_AUC.png", ()),
                                        ("hits", f"Freezing hours caught by top-1% alarms\n(of {int(d['test_events'].iloc[0])} h)",
                                         "b_alarm_hits.png", ()),
                                        ("rmse", "Outlet temperature RMSE [°C]", "c_temperature_RMSE.png", ("gbm",))):
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300, facecolor="white")
        ax.set_facecolor("white")
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for j, (m, label, color, marker) in enumerate(MODELS):
            if m in skip:
                continue
            r = s.xs(m, level="model").reindex(order)
            mean, sd = r[f"{metric}_mean"].to_numpy(float), r[f"{metric}_sd"].fillna(0).to_numpy(float)
            xx = x + (j - 1) * 0.14
            ax.errorbar(xx, mean, yerr=sd, color=color, lw=2, marker=marker, ms=8, mec="white", mew=1.5,
                        capsize=4, label=label)
            for xi, v in zip(xx, mean):
                if np.isfinite(v):
                    off = {"softw": (-14, 12), "nn": (-4, -18), "gbm": (0, -20)}[m]
                    ax.annotate(f"{v:.3f}" if metric != "hits" else f"{v:.0f}", (xi, v),
                                xytext=off, textcoords="offset points",
                                ha="center", fontsize=9.5, color=color)
        ax.margins(y=0.12)
        ax.set_xticks(x, [tick(y) for y in order])
        ax.set_xlabel("Training data (most recent years before validation)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=3, fontsize=11)
        ax.text(1.0, 1.005, "Neural models: mean ± SD over 5 seeds", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, color=MUTED)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=300, facecolor="white")
        plt.close(fig)
        print("saved:", fname)


if __name__ == "__main__":
    main()
