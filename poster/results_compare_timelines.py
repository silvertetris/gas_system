"""Research Results — comparison timelines against the final KF–soft PINN, one panel per file (line M, English).

Pairs: KF–soft PINN (baseline) vs  AE–soft PINN · KF–hard PINN · Neural network (no physics) · Gradient boosting.
Per pair: a temperature (test period) · b freezing probability (test period) · c temperature (10 days) ·
d freezing probability (10 days). Gradient boosting has no temperature output → b·d only.
Numbers come from ttttt/5_모델비교/model_accuracy_min.csv (line M); alarm hits are re-computed and checked against it.

    PYTHONPATH=. .venv/bin/python poster/results_compare_timelines.py   → ttttt/poster_graphs/results_compare/
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poster.results_model_timelines import (ACC, AXIS, INK, INK2, LAST_VALID, MODELS, OPT, ORANGE, ROOT, ZOOM,
                                            alarms, new_ax)

OUT = ROOT / "ttttt" / "poster_graphs" / "results_compare"
TRAIN = "M"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)
    print("saved:", name)


def load():
    acc = pd.read_csv(ACC, encoding="utf-8-sig")
    acc = acc[acc["계열"] == TRAIN].set_index(["정제", "모델"])
    info = {}
    for key, label, d, pcol, color, ref, csv_name in MODELS:
        pred = pd.read_parquet(OPT / d / f"pred_{TRAIN}.parquet").sort_index()
        ev = (pred["ev"] > 0.5).to_numpy()
        k = int(np.ceil(0.01 * len(pred)))
        hits = int((alarms(pred, pcol, k) & ev).sum())
        r = acc.loc[(ref, csv_name)]
        assert hits == int(r["경보가 잡은 빙결"]), f"{label}: alarm hits {hits} ≠ table"
        info[key] = {"label": label, "pred": pred, "p": pred[pcol], "color": color, "has_temp": pcol == "p",
                     "thr": float(np.sort(pred[pcol].to_numpy())[::-1][k - 1]), "hits": hits,
                     "events": int(ev.sum()), "auc": str(r["시험 AUC"]), "rmse": str(r["공급온도 RMSE ℃"])}
    return info


def date_axis(ax, zoom):
    if zoom:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = load()
    base_key = MODELS[0][0]
    B = info[base_key]
    full = pd.date_range(B["pred"].index.min(), LAST_VALID, freq="1h")
    zoom = pd.date_range(ZOOM[0], ZOOM[1], freq="1h")

    t_all = pd.concat([i["pred"].loc[:LAST_VALID, "mu"] for i in info.values() if i["has_temp"]] +
                      [B["pred"].loc[:LAST_VALID, "y"]])
    t_lo, t_hi = min(-4.0, float(t_all.min()) - 1), float(t_all.max()) + 2
    z_all = pd.concat([i["pred"].loc[ZOOM[0]:ZOOM[1], ["mu", "y"]].stack() for i in info.values() if i["has_temp"]])
    z_lo, z_hi = float(z_all.min()) - 0.8, float(z_all.max()) + 0.8

    for n, (key, *_rest) in enumerate(MODELS[1:], start=2):
        C = info[key]
        tag = f"{n}_KF-soft-PINN_vs_{key.split('_', 1)[1]}"

        for zoomed, idx, sfx_t, sfx_p in ((False, full, "a_temperature_test_period", "b_probability_test_period"),
                                           (True, zoom, "c_temperature_10days", "d_probability_10days")):
            bw, cw = B["pred"].reindex(idx), C["pred"].reindex(idx)
            width = 12 if zoomed else 14
            lw = 1.3 if zoomed else 0.55

            if C["has_temp"]:
                fig, ax = new_ax(width)
                ax.plot(bw.index, bw["y"], color=INK2, lw=0.9 if zoomed else 0.5, label="Actual")
                ax.plot(cw.index, cw["mu"], color=C["color"], lw=lw, label=C["label"])
                ax.plot(bw.index, bw["mu"], color=B["color"], lw=lw, label=B["label"])
                e = bw[bw["ev"] > 0.5]
                ax.scatter(e.index, e["y"], s=16 if zoomed else 7, color=ORANGE, lw=0, zorder=4,
                           label="Freezing (< 0 °C)")
                ax.axhline(0, color=AXIS, lw=1)
                if zoomed:
                    ax.set_ylim(z_lo, z_hi)
                else:
                    ax.set_ylim(t_lo, t_hi)
                    ax.text(0.995, 0.97, f"Temperature RMSE: {B['label']} {B['rmse']} °C  ·  {C['label']} {C['rmse']} °C",
                            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=INK2)
                ax.set_ylabel("Min. outlet gas temp.\nin next 6 h [°C]")
                date_axis(ax, zoomed)
                ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.22), markerscale=2)
                save(fig, f"{tag}_{sfx_t}.png")

            fig, ax = new_ax(width)
            ax.plot(idx, C["p"].reindex(idx), color=C["color"], lw=lw, label=C["label"])
            ax.plot(idx, B["p"].reindex(idx), color=B["color"], lw=lw, label=B["label"])
            ax.axhline(C["thr"], color=C["color"], lw=1.1, ls=(0, (5, 4)), alpha=0.8)
            ax.axhline(B["thr"], color=B["color"], lw=1.1, ls=(0, (5, 4)), alpha=0.8)
            ax.plot([], [], color=INK2, lw=1.1, ls=(0, (5, 4)), label="Top-1% alarm threshold (each model)")
            ax.vlines(idx[bw["ev"].to_numpy() > 0.5], -0.09, -0.03, color=INK, lw=0.9 if zoomed else 0.7,
                      label="Actual freezing")
            ax.set_ylim(-0.11, 1.05 if zoomed else 1.45)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_ylabel("Freezing probability\nwithin 6 h")
            if not zoomed:
                for j, M_ in enumerate((B, C)):
                    ax.text(0.005, 0.99 - j * 0.085,
                            f"{M_['label']}: test AUC {M_['auc']}  ·  top-1% alarms caught {M_['hits']} of "
                            f"{M_['events']} h", transform=ax.transAxes, va="top", fontsize=10.5, color=M_["color"])
            date_axis(ax, zoomed)
            ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.22))
            save(fig, f"{tag}_{sfx_p}.png")


if __name__ == "__main__":
    main()
