"""Gas supply system schematic — English, component names only, transparent background (poster PNG, 300 dpi).

Natural gas inlet → inlet header → gas heaters A/B + bypass valve → mixing header → pressure regulator
→ downstream pipe → metering point (prediction target). Dashed box = freeze-risk zone.

    .venv/bin/python poster/diagram_gas_system.py      # → ttttt/Gas_system_diagram.png
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ttttt" / "Gas_system_diagram.png"
FONT = "DejaVu Sans"
INK2 = "#52514e"
C = {  # (edge, fill)
    "gas": ("#898781", "#f0efec"), "heat": ("#eb6834", "#fde9df"), "reg": ("#2a78d6", "#e3effc"),
    "out": ("#c98500", "#fff4d6"),
}


def box(ax, x, yc, w, h, kind, title, size=14):
    edge, fill = C[kind]
    ax.add_patch(FancyBboxPatch((x, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fill, ec=edge,
                                lw=2.2))
    ax.text(x + w / 2, yc, title, ha="center", va="center", fontsize=size, fontweight="bold", color=edge, family=FONT,
            linespacing=1.25)


def arrow(ax, p, q, color=INK2, lw=2.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=20, color=color, lw=lw, shrinkA=0, shrinkB=0))


def line(ax, xs, ys, color=INK2, lw=2.4):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round")


def main():
    fig = plt.figure(figsize=(18, 5.6), dpi=300, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 56); ax.axis("off")

    Y, H = 28, 11          # main-line centre, box height
    ya, yb = 46, 10        # heater A / bypass centres

    box(ax, 2, Y, 18, H, "gas", "Natural gas\ninlet")
    box(ax, 26, Y, 14, H, "gas", "Inlet\nheader")
    box(ax, 48, ya, 22, H, "heat", "Gas heater A")
    box(ax, 48, Y, 22, H, "heat", "Gas heater B")
    box(ax, 48, yb, 22, H, "gas", "Bypass valve")
    box(ax, 78, Y, 16, H, "gas", "Mixing\nheader")
    box(ax, 101, Y, 20, H, "reg", "Pressure\nregulator")
    box(ax, 127, Y, 20, H, "gas", "Downstream\npipe")
    box(ax, 153, Y, 18, H, "out", "Metering\npoint")

    arrow(ax, (20, Y), (26, Y))
    # split: inlet header → heaters / bypass
    line(ax, [40, 44], [Y, Y]); line(ax, [44, 44], [yb, ya])
    for y in (ya, Y, yb):
        arrow(ax, (44, y), (48, y))
    # merge: heaters / bypass → mixing header
    for y in (ya, Y, yb):
        line(ax, [70, 74], [y, y])
    line(ax, [74, 74], [yb, ya]); arrow(ax, (74, Y), (78, Y))
    arrow(ax, (94, Y), (101, Y))
    arrow(ax, (121, Y), (127, Y))
    arrow(ax, (147, Y), (153, Y))
    arrow(ax, (171, Y), (178, Y))

    # freeze-risk zone
    edge = C["reg"][0]
    ax.add_patch(FancyBboxPatch((98, 19), 76, 22, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=edge,
                                lw=2.0, linestyle=(0, (5, 4))))
    ax.text(136, 37.5, "Freeze-risk zone", ha="center", va="center", fontsize=13, fontweight="bold", color=edge,
            family=FONT)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, transparent=True)
    plt.close(fig)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
