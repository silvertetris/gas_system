"""KF–soft PINN architecture diagram — English, box titles + arrows only, transparent background (poster PNG, 300 dpi).

Boxes are left blank below their titles so explanations can be added on the poster.
① Operating data → ② Kalman filter → ③ Neural network (temperature head · state head) → ④ Physics layer
→ ⑤ Loss (training) → ⑥ Output / attribution. Solid = prediction path, dashed = training-loss path.

    .venv/bin/python poster/diagram_kf_softpinn_en.py      # → ttttt/KF_softPINN_diagram.png
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ttttt" / "KF_softPINN_diagram.png"
FONT = "DejaVu Sans"
INK2 = "#52514e"
C = {  # (edge, fill)
    "data": ("#898781", "#f0efec"), "kf": ("#1baf7a", "#e3f4ee"), "nn": ("#2a78d6", "#e3effc"),
    "phys": ("#eb6834", "#fde9df"), "loss": ("#4a3aa7", "#ece9f8"), "out": ("#c98500", "#fff4d6"),
}
TITLE_GAP = 2.6  # box top → title centre


def box(ax, x, y, w, h, kind, title, size=16, r=1.2, lw=2.2):
    edge, fill = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - TITLE_GAP, title, ha="center", va="center", fontsize=size, fontweight="bold",
            color=edge, family=FONT)


def arrow(ax, p, q, color=INK2, dashed=False, rad=0.0, lw=2.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, color=color, lw=lw,
                                 linestyle=(0, (5, 4)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def main():
    fig = plt.figure(figsize=(18, 4.6), dpi=300, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 46); ax.axis("off")

    box(ax, 2, 6, 30, 39, "data", "① Operating data", 15)
    box(ax, 40, 28, 30, 15, "kf", "② Kalman filter")
    box(ax, 78, 34.5, 30, 9, "nn", "Temperature head", 14.5)
    box(ax, 78, 21, 30, 9.5, "nn", "③ Neural network")
    box(ax, 78, 6, 30, 11, "nn", "State head", 14.5)
    box(ax, 116, 38.5, 62, 6.5, "out", "⑥ Output")
    box(ax, 116, 24, 62, 11.5, "loss", "⑤ Loss function (training)")
    box(ax, 116, 6, 34, 14, "phys", "④ Physics layer")
    box(ax, 156, 6, 22, 14, "out", "Attribution", 14.5)

    # prediction path (solid)
    arrow(ax, (32, 35.5), (40, 35.5))          # data → KF
    arrow(ax, (32, 25.5), (78, 25.5))              # data → NN
    arrow(ax, (70, 30), (78, 27))            # KF → NN
    arrow(ax, (93, 30.5), (93, 34.5))              # NN → temperature head
    arrow(ax, (93, 21), (93, 17))              # NN → state head
    arrow(ax, (108, 11.5), (116, 11.5))            # state head → physics
    arrow(ax, (108, 41.5), (116, 41.5))            # temperature head → output
    arrow(ax, (150, 13), (156, 13))        # physics → attribution
    # training-loss path (dashed)
    arrow(ax, (133, 20), (133, 24), color=C["loss"][0], dashed=True)          # physics → loss
    arrow(ax, (108, 37), (116, 32), color=C["loss"][0], dashed=True)          # temperature head → loss
    arrow(ax, (17, 6), (120, 6.15), color=C["phys"][0], dashed=True, rad=0.06)  # observed states → physics

    # legend
    ax.add_patch(FancyArrowPatch((124, 1.8), (132, 1.8), arrowstyle="-|>", mutation_scale=18, color=INK2, lw=2.2))
    ax.text(133.5, 1.8, "Prediction", va="center", fontsize=12.5, color=INK2, family=FONT)
    ax.add_patch(FancyArrowPatch((150, 1.8), (158, 1.8), arrowstyle="-|>", mutation_scale=18, color=C["loss"][0],
                                 lw=2.2, linestyle=(0, (5, 4))))
    ax.text(159.5, 1.8, "Training loss", va="center", fontsize=12.5, color=INK2, family=FONT)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, transparent=True)
    plt.close(fig)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
