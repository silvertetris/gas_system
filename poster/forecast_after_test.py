"""Final KF–soft PINN run up to the last valid measurement (line M) — English poster graph + CSV.

The model forecasts freezing risk for the NEXT 6 HOURS from current measurements; it has no weather/operation
forecast, so it cannot run further ahead than the last valid measurement + 6 h.

⚠ Data check (2026-09-16): SCADA logging effectively stops on 2026-04-20 14:00. After that only scattered records exist
and every sensor repeats a frozen value (daily distinct values ≤ 2, SD 0), e.g. outlet temperature 7.714 °C all
summer. Those hours are flagged `valid_input = False` and are not plotted as predictions.

Steps
  1. Rebuild the exact training-time input scaling (same split, same KF refinement file, same missing flags).
  2. Refit the isotonic calibrator exactly as the pipeline did (seed-42 model, validation block) — it was not saved.
  3. Check: reproduced test predictions must equal the saved pred_M.parquet (else abort).
  4. Predict every hour with complete core inputs from PLOT_FROM; mark hours after the last valid measurement.

    PYTHONPATH=. .venv/bin/python poster/forecast_after_test.py
      → ttttt/poster_graphs/18_forecast_last_measurement.png · 18_forecast_last_measurement.csv
"""
from __future__ import annotations

import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import IsotonicRegression

from pinn.restricted import config, data, folds, model as M, tune

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "pinn" / "restricted" / "output" / "optuna" / "softw_min" / "v_KF"
OUT = ROOT / "ttttt" / "poster_graphs"
TRAIN, TARGET, DEV = "M", "min", "cpu"
PLOT_FROM = "2026-03-01"
CORE = ["t61", "t61_min", "hdr", "t_in", "dp"]
MIN_RECORDS_PER_HOUR = 30         # a live hour has ~60 1-min records
MIN_DISTINCT_PER_HOUR = 3         # a frozen logger repeats 1 value or flips between 2

INK2, MUTED, AXIS, GRID = "#52514e", "#898781", "#c3c2b7", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "legend.labelcolor": INK2})


def last_valid_time() -> pd.Timestamp:
    spec = config.TRAINS[TRAIN]
    g = pd.read_parquet(config.TREND_PARQUET, columns=["Time", spec["hdr"], spec["t_in"]]).set_index("Time")
    g = g.sort_index().loc[PLOT_FROM:]
    hr = g.groupby(g.index.floor("h"))
    live = (hr.size() >= MIN_RECORDS_PER_HOUR) & (hr.nunique().min(axis=1) >= MIN_DISTINCT_PER_HOUR)
    return live.index[live][-1]                                   # start of the last fully logged hour


def main():
    ck = torch.load(RUN / f"model_{TRAIN}.pt", map_location=DEV, weights_only=False)
    hp, cols, names, refine = ck["hp"], ck["cols"], ck["names"], tuple(ck["refine"])
    net = M.make(ck["arch"], ck["n_feat"], ck["ua_init_kw"], hp["hidden"], hp["n_layers"], hp["dropout"],
                 ck["dip"]).to(DEV)
    net.load_state_dict(ck["state"])
    net.eval()

    # --- 1. same frame / split / scaling as optuna_pipeline.run_train
    Xb, _ = folds.base_frame(TRAIN, TARGET)
    dev_m, test_m, _ = folds.dev_test(Xb)
    in_tr, in_va, _ = folds.inner_split(Xb[dev_m])
    ref = folds.load_refinements(TRAIN, ["final"])["final"]
    X = tune.frame(Xb, ref, refine)
    pos = np.flatnonzero(dev_m)
    tr_all = np.zeros(len(X), bool); tr_all[pos[in_tr]] = True
    va_all = np.zeros(len(X), bool); va_all[pos[in_va]] = True

    A = X[cols].to_numpy(np.float64)
    mu_, sd_ = np.nanmean(A[tr_all], 0), np.nanstd(A[tr_all], 0)
    sd_[~np.isfinite(sd_) | (sd_ < 1e-9)] = 1.0
    mu_[~np.isfinite(mu_)] = 0.0
    flag_idx = [cols.index(n[len("결측_"):]) for n in names if n.startswith("결측_")]
    assert names == list(cols) + [f"결측_{cols[i]}" for i in flag_idx], "feature layout mismatch"

    def features(df: pd.DataFrame) -> torch.Tensor:
        z = (df[cols].to_numpy(np.float64) - mu_) / sd_
        miss = ~np.isfinite(z)
        z = np.hstack([np.where(miss, 0.0, z), miss[:, flag_idx]]).astype(np.float32)
        return torch.tensor(z, device=DEV)

    @torch.no_grad()
    def predict(df: pd.DataFrame):
        z = features(df)
        mu, sd = net(z)[:2]
        return net.prob_freeze(z).cpu().numpy(), mu.cpu().numpy(), sd.cpu().numpy()

    # --- 2. calibrator (same data as the pipeline)
    ev = (X["y_t61_min"] < config.FREEZE_C).to_numpy(float)
    p_va, _, _ = predict(X[va_all])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_va, ev[va_all])

    # --- 3. reproduction check against the saved test predictions
    saved = pd.read_parquet(RUN / f"pred_{TRAIN}.parquet").sort_index()
    p_te_raw, mu_te, _ = predict(X[test_m])
    p_te = np.clip(iso.predict(p_te_raw), 0, 1)
    dp_, dmu = np.abs(p_te - saved["p"].to_numpy()).max(), np.abs(mu_te - saved["mu"].to_numpy()).max()
    print(f"reproduction check: max |Δp| {dp_:.2e} · max |Δμ| {dmu:.2e} °C over {test_m.sum()} test hours")
    assert dp_ < 1e-4 and dmu < 1e-3, "could not reproduce saved predictions — aborting"
    last_test = X.index[test_m][-1]
    last_valid = last_valid_time()
    stale_test = saved.index > last_valid
    print(f"last valid measurement {last_valid} · test period ends {last_test} · test hours on frozen data "
          f"{int(stale_test.sum())} (freezing among them {int(saved.loc[stale_test, 'ev'].sum())})")

    # --- 4. every hour with complete core inputs (labels not required)
    f = data.build(TRAIN, refine=(), target=TARGET)
    f = f[(f.index >= pd.Timestamp(PLOT_FROM)) & f[CORE].notna().all(axis=1)]
    f = tune.frame(f, ref, refine)
    p_raw, mu, sd = predict(f)
    out = pd.DataFrame({
        "outlet_temp_now": f["t61"], "p_freeze_6h": np.clip(iso.predict(p_raw), 0, 1),
        "pred_min_outlet_temp_6h": mu, "pred_sd": sd, "actual_min_outlet_temp_6h": f["y_t61_min"],
        "valid_input": f.index <= last_valid}, index=f.index)
    k = int(np.ceil(0.01 * len(saved)))
    thr = float(np.sort(saved["p"].to_numpy())[::-1][k - 1])
    out["top1pct_alarm"] = out["p_freeze_6h"] >= thr
    out.to_csv(OUT / "18_forecast_last_measurement.csv", float_format="%.4f")

    v = out[out["valid_input"]]
    last = v.iloc[-1]
    t0 = v.index[-1]
    print(f"FORECAST from last valid measurement {t0} (window {t0 + pd.Timedelta(hours=1)} → "
          f"{t0 + pd.Timedelta(hours=6, minutes=59)}): outlet temp now {last['outlet_temp_now']:.2f} °C · "
          f"predicted min {last['pred_min_outlet_temp_6h']:.2f} ± {last['pred_sd']:.2f} °C · "
          f"P(freeze) {last['p_freeze_6h']:.4f} (alarm threshold {thr:.4f}) · actual min {last['actual_min_outlet_temp_6h']:.2f}")
    print(f"hours with inputs after last valid measurement (frozen values, not used): {int((~out['valid_input']).sum())}")

    # --- graph: valid period only, frozen-logger stretch shaded
    end = out.index[-1] + pd.Timedelta(days=2)
    w = v.reindex(pd.date_range(v.index.min(), v.index.max(), freq="1h"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6.8), dpi=300, sharex=True, facecolor="white",
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    for ax in (a1, a2):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvspan(last_valid, end, color="#f3f2ee", lw=0, zorder=0)
    a1.plot(w.index, w["actual_min_outlet_temp_6h"], color=INK2, lw=0.7, label="Actual")
    a1.plot(w.index, w["pred_min_outlet_temp_6h"], color=BLUE, lw=0.8, label="Predicted (KF–soft PINN)")
    a1.errorbar([t0], [last["pred_min_outlet_temp_6h"]], yerr=[last["pred_sd"]], fmt="none", ecolor=ORANGE,
                elinewidth=1.5, capsize=4)
    a1.scatter([t0], [last["pred_min_outlet_temp_6h"]], marker="*", s=280, color=ORANGE, zorder=5,
               label="6-h forecast from last valid measurement (±1σ)")
    a1.axhline(0, color=ORANGE, lw=1.2, ls=(0, (5, 4)))
    ymax = float(np.nanmax(v[["pred_min_outlet_temp_6h", "actual_min_outlet_temp_6h"]].to_numpy())) + 2
    a1.set_ylim(min(-3.0, float(np.nanmin(v["pred_min_outlet_temp_6h"])) - 1), ymax)
    a1.text(last_valid + pd.Timedelta(days=4), ymax - 0.6, "No valid measurements\n(logger values frozen)",
            va="top", fontsize=11, color=MUTED)
    a1.set_ylabel("Min. outlet gas temperature\nin next 6 h [°C]")
    a1.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.2))

    a2.plot(w.index, w["p_freeze_6h"], color=BLUE, lw=0.9, label="Freezing probability")
    a2.scatter([t0], [last["p_freeze_6h"]], marker="*", s=280, color=ORANGE, zorder=5)
    a2.axhline(thr, color=MUTED, lw=1.2, ls=(0, (5, 4)), label=f"Top-1% alarm threshold ({thr:.3f})")
    a2.set_ylim(-0.03, 1.03)
    a2.set_ylabel("Freezing probability\nwithin 6 h")
    a2.legend(loc="upper right")
    a2.set_xlim(pd.Timestamp(PLOT_FROM), end)
    a2.xaxis.set_major_locator(mdates.MonthLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(OUT / "18_forecast_last_measurement.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved: 18_forecast_last_measurement.png / .csv")


if __name__ == "__main__":
    main()
