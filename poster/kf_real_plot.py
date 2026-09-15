"""Real Kalman-filter result on the supply-gas temperature (train M, 1-min data) — English poster figure.

Uses exactly the pipeline filter (`risk.refine.kf_innovation`, steady-state gain 0.05, reset at segment/mode blocks),
run on the full 1-min series, then shows a 48-hour window around the densest freeze hours of the test period.

    PYTHONPATH=. .venv/bin/python poster/kf_real_plot.py      # → ttttt/KF_filtering_M.png
"""
from __future__ import annotations

import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk import modes, config as rconfig
from risk.refine import kf_innovation, KF_GAIN

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ttttt" / "KF_filtering_M.png"
GRAY, AQUA, INK, INK2, AXIS, GRID = "#898781", "#1baf7a", "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"


def main():
    g = modes.minute_frame("M")
    spec = rconfig.TRAINS["M"]
    blk = pd.factorize((g["segment_id"].astype(str) + "|" + g["mode"].astype(str)).to_numpy())[0]
    y = g[spec["t61"]].to_numpy(float)
    y = np.where(y <= -29.5, np.nan, y)                      # sensor lower-limit readings removed (as in the target)
    lvl, inn = kf_innovation(y, blk)
    s = pd.DataFrame({"z": y, "x": lvl, "nu": inn}, index=g.index)

    # 48 h window with the most sub-zero minutes in the test period
    test = s.loc["2023-08-03":]
    neg = (test["z"] < 0).astype(float).rolling("48h").sum()
    end = neg.idxmax()
    w = s.loc[end - pd.Timedelta(hours=48): end]
    print("window:", w.index[0], "→", w.index[-1], "| sub-zero minutes:", int((w["z"] < 0).sum()),
          "| noise std raw-KF:", round(float(np.nanstd(w["z"] - w["x"])), 3))

    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=300, facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(w.index, w["z"], s=4, color=GRAY, alpha=0.55, lw=0, label=r"Measurement $z_k$")
    ax.plot(w.index, w["x"], color=AQUA, lw=2.2, label=r"KF estimate $\hat{x}_k$")
    ax.axhline(0, color=AXIS, lw=1.2)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=11)
    ax.set_ylabel("Supply gas temperature [°C]", fontsize=12, color=INK2)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2, frameon=False, fontsize=12, labelcolor=INK2)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")             # 그래프는 흰 배경
    plt.close(fig)
    print("saved:", OUT, "| gain", KF_GAIN)


if __name__ == "__main__":
    main()
