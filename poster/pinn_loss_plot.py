"""Final KF–soft PINN training loss curve — English, white background (poster PNG, 300 dpi).

Reads the saved training history of the final model (softw · min target · KF · line M) and plots training /
validation loss per epoch on a log axis. Seed 42 is drawn solid; the other four seeds' validation curves faint. Epoch 0 (random init) is omitted.

    .venv/bin/python poster/pinn_loss_plot.py      # → ttttt/PINN_loss_curve_M.png
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "pinn" / "restricted" / "output" / "optuna" / "softw_min" / "v_KF" / "history_M.csv"
OUT = ROOT / "ttttt" / "PINN_loss_curve_M.png"
FONT = "DejaVu Sans"
INK2, GRID, AXIS = "#52514e", "#e4e3df", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MAIN_SEED = 42


def main():
    h = pd.read_csv(SRC, encoding="utf-8-sig").rename(columns={"시드": "seed"})
    h = h[h["epoch"] >= 1]  # epoch 0 = random-init training loss (~430), off-scale
    plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=300, facecolor="white")
    ax.set_facecolor("white")

    for s, g in h.groupby("seed"):
        if s != MAIN_SEED:
            ax.plot(g["epoch"], g["valid"], color=BLUE, lw=1.2, alpha=0.25)
    g = h[h["seed"] == MAIN_SEED]
    ax.plot(g["epoch"], g["train"], color=ORANGE, lw=2.2, label="Training loss")
    ax.plot(g["epoch"], g["valid"], color=BLUE, lw=2.2, label="Validation loss")
    ax.plot([], [], color=BLUE, lw=1.2, alpha=0.25, label="Validation loss (other seeds)")
    best = g.loc[g["valid"].idxmin()]
    ax.plot(best["epoch"], best["valid"], "o", ms=8, color=BLUE, mec="white", mew=2, zorder=5)
    ax.annotate(f"Best epoch {int(best['epoch'])}", (best["epoch"], best["valid"]), xytext=(0, 16),
                textcoords="offset points", ha="center", fontsize=11, color=INK2)

    ax.set_yscale("log")
    ax.set_yticks([3, 4, 5, 6, 8, 10]); ax.set_yticks([], minor=True)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    ax.set_xlabel("Epoch", fontsize=13, color=INK2)
    ax.set_ylabel("Loss (log scale)", fontsize=13, color=INK2)
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(axis="y", which="major", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.legend(loc="upper right", frameon=False, fontsize=12, labelcolor=INK2)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print("saved:", OUT, "| best", dict(best[["seed", "epoch", "valid"]]))


if __name__ == "__main__":
    main()
