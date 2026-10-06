"""Outlet gas temperature trend (prediction-target sensor, line M) — English, white background (poster PNG, 300 dpi).

Raw 1-min readings with sensor lower-limit values (≤ −29.5 °C) removed, as in the model target.
  16_outlet_temperature_trend.png   daily mean + daily minimum, 2013-10 → end, test period shaded (axis −10…32 °C)
  17_outlet_temperature_winter.png  hourly minimum for the coldest test winter

    PYTHONPATH=. .venv/bin/python poster/outlet_temp_trend.py
"""
from __future__ import annotations

import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk import modes, config as rconfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ttttt" / "poster_graphs"
START, TEST_START = "2013-10-01", "2023-08-03"
INK2, MUTED, AXIS, GRID = "#52514e", "#898781", "#c3c2b7", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "legend.labelcolor": INK2})


def style(ax):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    g = modes.minute_frame("M")
    y = g[rconfig.TRAINS["M"]["t61"]].astype(float)
    y = y.where(y > -29.5).loc[START:]
    print("range:", y.first_valid_index(), "→", y.last_valid_index(), "| valid minutes:", int(y.notna().sum()))

    d = y.resample("D").agg(["mean", "min", "max", "count"])
    d = d[d["count"] >= 60]                                     # days with at least 1 h of readings
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))

    fig, ax = plt.subplots(figsize=(14, 4.6), dpi=300, facecolor="white")
    style(ax)
    ax.axvspan(pd.Timestamp(TEST_START), d.index.max(), color="#f3f2ee", lw=0, zorder=0)
    lo, hi = -10.0, 32.0
    n_out = int(((d["mean"] > hi) | (d["min"] < lo)).sum())
    print("days outside axis (sensor faults):", n_out)
    ax.plot(d.index, d["min"], color=MUTED, lw=0.6, alpha=0.9, label="Daily minimum")
    ax.plot(d.index, d["mean"], color=BLUE, lw=1.0, label="Daily mean")
    ax.axhline(0, color=ORANGE, lw=1.4, ls=(0, (5, 4)), label="0 °C (freezing)")
    ax.set_ylim(lo, hi)
    ax.text(pd.Timestamp(TEST_START) + pd.Timedelta(days=40), hi - 1, "Test period", va="top", fontsize=12,
            color=MUTED)
    if n_out:
        ax.text(0.995, 0.02, f"{n_out} sensor-fault days outside the axis not shown", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, color=MUTED)
    ax.set_ylabel("Outlet gas temperature [°C]")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(d.index.min(), d.index.max())
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(OUT / "16_outlet_temperature_trend.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved: 16_outlet_temperature_trend.png")

    # coldest winter (Nov–Mar) in the test period, by sub-zero minutes
    test = y.loc[TEST_START:]
    winters = {yr: int((test.loc[f"{yr}-11-01":f"{yr + 1}-03-31"] < 0).sum())
               for yr in range(test.index[0].year, test.index[-1].year)}
    yr = max(winters, key=winters.get)
    print("sub-zero minutes by winter:", winters, "→", yr)
    w = y.loc[f"{yr}-11-01":f"{yr + 1}-03-31"]
    h = w.resample("h").agg(["mean", "min"])
    fig, ax = plt.subplots(figsize=(14, 4.6), dpi=300, facecolor="white")
    style(ax)
    ax.plot(h.index, h["min"], color=BLUE, lw=0.8, label="Hourly minimum")
    neg = h[h["min"] < 0]
    ax.scatter(neg.index, neg["min"], s=10, color=ORANGE, zorder=4, lw=0, label="Below 0 °C")
    ax.axhline(0, color=ORANGE, lw=1.4, ls=(0, (5, 4)))
    ax.set_ylabel("Outlet gas temperature [°C]")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(h.index.min(), h.index.max())
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.14), markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / "17_outlet_temperature_winter.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved: 17_outlet_temperature_winter.png")


if __name__ == "__main__":
    main()
