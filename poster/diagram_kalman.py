"""Kalman filter concept diagram (poster PNG, 300 dpi) — equations and keywords only, English.

Initial state → Predict ↔ Update loop · measurement z_k in · estimate x̂_k out.

    .venv/bin/python poster/diagram_kalman.py      # → ttttt/Kalman_filter_diagram.png
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ttttt" / "Kalman_filter_diagram.png"
FONT = "DejaVu Sans"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
BLUE, ORANGE, AQUA, GRAY = ("#2a78d6", "#e3effc"), ("#eb6834", "#fde9df"), ("#1baf7a", "#e3f4ee"), ("#898781", "#f0efec")


def box(ax, x, y, w, h, col, lw=2.6, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=col[1], ec=col[0], lw=lw))


def txt(ax, x, y, s, size=16, color=INK, bold=False, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, family=FONT, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color=INK2, rad=0.0, lw=2.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=26, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def main():
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig = plt.figure(figsize=(16, 6.4), dpi=300, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 64); ax.axis("off")

    # Initial state
    box(ax, 3, 30, 18, 12, GRAY, lw=2.0, r=1.0)
    txt(ax, 12, 38.5, "Initial state", 14, INK2, bold=True)
    txt(ax, 12, 33.8, r"$\hat{x}_0,\ P_0$", 17)
    arrow(ax, (21, 36), (29, 36))

    # Predict
    box(ax, 29, 18, 44, 36, BLUE)
    txt(ax, 51, 49, "① Predict", 20, BLUE[0], bold=True)
    txt(ax, 51, 38, r"$\hat{x}_k^{-} = A\,\hat{x}_{k-1} + B\,u_k$", 20)
    txt(ax, 51, 27, r"$P_k^{-} = A\,P_{k-1}\,A^{T} + Q$", 20)

    # Update
    box(ax, 89, 12, 50, 42, ORANGE)
    txt(ax, 114, 49, "② Update", 20, ORANGE[0], bold=True)
    txt(ax, 114, 40, r"$K_k = P_k^{-}H^{T}\,(H\,P_k^{-}H^{T} + R)^{-1}$", 19)
    txt(ax, 114, 29.5, r"$\hat{x}_k = \hat{x}_k^{-} + K_k\,(z_k - H\,\hat{x}_k^{-})$", 19)
    txt(ax, 114, 19, r"$P_k = (I - K_k H)\,P_k^{-}$", 19)

    # Predict → Update
    arrow(ax, (73, 36), (89, 36))
    txt(ax, 81, 40, r"$\hat{x}_k^{-},\ P_k^{-}$", 15, INK2)

    # Measurement in
    box(ax, 99, 56.5, 30, 7, GRAY, lw=2.0, r=1.0)
    txt(ax, 114, 60, r"Measurement  $z_k$", 16, INK, bold=True)
    arrow(ax, (114, 56.5), (114, 54))

    # Estimate out
    box(ax, 143, 27, 15, 18, AQUA, lw=2.2, r=1.0)
    txt(ax, 150.5, 39, "Estimate", 15, AQUA[0], bold=True)
    txt(ax, 150.5, 32.5, r"$\hat{x}_k$", 20)
    arrow(ax, (139, 36), (143, 36))

    # Loop k → k+1
    arrow(ax, (114, 12), (51, 18), rad=-0.35, color=INK2)
    txt(ax, 82, 9.5, r"$k \rightarrow k+1$", 17, INK2)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, transparent=True)             # 배경 투명
    plt.close(fig)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
