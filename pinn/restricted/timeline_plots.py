"""빙결 예측 시계열 — 시험구간 전체 (조합·계열당 한 장).

시험구간(개발에 쓰지 않은 마지막 20%)만 그린다. 개발구간 예측은 학습에 쓴 데이터라 좋게 보이는 게 당연해
시계열로 보여주면 오해를 부른다.

패널 (시간축 공유, 지표마다 축 하나):
  ① 공급온도 — 실제 6h 내 최저 T61 vs 예측 평균 ±1σ, 빙결 사건 표시. 센서 하한(−30℃) 급락은 축 밖 표기
  ② 빙결 확률 — PINN vs 같은 입력 GBM (시각 t 에서 "앞으로 6h 안에 T61 < 0" 확률) + PINN 상위 1% 경보선,
     실제 빙결 시각은 아래 눈금
  ③ 월별 — 빙결 사건 시간 수 · 그중 상위 1% 경보가 잡은 시간 수 (PINN / GBM)
  ④ 확대 — 빙결 시간이 가장 많았던 10일: 공급온도 · ⑤ 같은 기간 확률
색: PINN blue `#2a78d6` · GBM orange `#eb6834` (검증 통과 쌍), 실제값·사건은 잉크색, 경보선은 muted.
표 원자료: pred_{계열}.parquet (시드 42 모델).
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import style
from . import config

log = logging.getLogger(__name__)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
LEG = dict(fontsize=8, frameon=False, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5,
           borderaxespad=0.2)


def _legend(ax):
    """범례 선은 굵게 — 본문 선(0.6~0.8)이 가늘어 범례에서 색이 안 보였다."""
    leg = ax.legend(**LEG)
    for h in leg.legend_handles:
        if hasattr(h, "set_linewidth"):
            h.set_linewidth(2.2)
    return leg
Y_FLOOR = -12.0        # 이보다 낮은 실제값(센서 하한 급락)은 축 밖으로 두고 개수만 적는다


def _axes(ax, grid="both"):
    ax.set_facecolor(SURF)
    ax.grid(True, axis=grid, color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _temp(ax, w, title):
    ax.fill_between(w.index, w["mu"] - w["sd"], w["mu"] + w["sd"], color=BLUE, alpha=0.15, lw=0, label="예측 ±1σ")
    ax.plot(w.index, w["y"].clip(lower=Y_FLOOR), color=INK2, lw=0.6, label="실제 6h 내 최저 T61")
    ax.plot(w.index, w["mu"], color=BLUE, lw=0.8, label="PINN 예측 평균")
    e = w[w["ev"] > 0.5]
    ax.scatter(e.index, e["y"].clip(lower=Y_FLOOR), s=6, color=INK, zorder=4, lw=0, label="빙결 (6h 내 < 0℃)")
    ax.axhline(config.FREEZE_C, color=AXIS, lw=1)
    n_out = int((w["y"] < Y_FLOOR).sum())
    if n_out:
        ax.text(0.995, 0.02, f"실제값 < {Y_FLOOR:.0f}℃ {n_out}시간 (센서 하한 급락 등) 은 {Y_FLOOR:.0f}℃ 에 표시",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    # 위쪽도 드문 튐(데이터가 드문 구간의 예측 폭주)에 끌려가지 않게 99.9% 분위로 자른다
    hi = float(np.nanquantile(np.r_[w["y"].to_numpy(), (w["mu"] + w["sd"]).to_numpy()], 0.999))
    n_hi = int(((w["mu"] > hi + 3) | (w["y"] > hi + 3)).sum())
    ax.set_ylim(Y_FLOOR - 1, hi + 3)
    if n_hi:
        ax.text(0.995, 0.98, f"축 위로 벗어난 {n_hi}시간 생략", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=MUTED)
    ax.set_ylabel("공급온도 T61 [℃]", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, pad=20)
    _legend(ax)


def _prob(ax, w, thr, title):
    ax.plot(w.index, w["p_gbm"], color=ORANGE, lw=0.7, label="GBM (같은 입력)")
    ax.plot(w.index, w["p"], color=BLUE, lw=0.8, label="PINN")
    ax.axhline(thr, color=MUTED, lw=1, label=f"PINN 상위 1% 분위 ({thr:.3f})")
    top = max(w["p"].max(), w["p_gbm"].max(), thr) * 1.1
    e = w.index[w["ev"] > 0.5]
    ax.vlines(e, -0.07 * top, -0.02 * top, color=INK, lw=0.6, label="실제 빙결 시각")
    ax.set_ylim(-0.08 * top, top)
    ax.set_ylabel("P(6h 내 빙결)", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, pad=20)
    _legend(ax)


def plot(out_dir, train: str):
    pred = pd.read_parquet(out_dir / f"pred_{train}.parquet").sort_index()
    ms = pd.read_csv(out_dir / f"metrics_seeds_{train}.csv")
    arch = ms["구조"].iloc[0] if "구조" in ms else "hard"
    target = ms["타깃"].iloc[0] if "타깃" in ms else "min_raw"
    r0 = ms.iloc[0]
    # 시간 격자를 채워 결측 구간을 선으로 잇지 않는다
    full = pd.date_range(pred.index.min(), pred.index.max(), freq=config.FREQ)
    w = pred.reindex(full)
    ev = pred["ev"] > 0.5
    thr = float(np.quantile(pred["p"], 0.99))
    # 경보 예산을 **같은 시간 수(상위 1%)** 로 맞춘다. 등장성 보정 확률은 계단이라 문턱값(≥ 분위수)으로 자르면
    # 동률이 몰려 PINN 경보가 1% 를 크게 넘었다(Z 552시간 = 2.6%). 동률은 PINN 은 예측 T61 평균이 낮은 순,
    # GBM 은 고정 난수 순으로 끊는다.
    k = int(np.ceil(0.01 * len(pred)))
    rng = np.random.default_rng(config.RANDOM_SEED)
    top_p = np.zeros(len(pred), bool); top_p[np.lexsort((pred["mu"].to_numpy(), -pred["p"].to_numpy()))[:k]] = True
    top_g = np.zeros(len(pred), bool); top_g[np.lexsort((rng.random(len(pred)), -pred["p_gbm"].to_numpy()))[:k]] = True

    fig = plt.figure(figsize=(20, 17.5), facecolor=SURF)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.0, 0.8, 1.0], hspace=0.62, wspace=0.12,
                          left=0.05, right=0.99, top=0.92, bottom=0.04)
    fig.suptitle(f"빙결 예측 시계열 — 계열 {train} · [{arch}·{target}] · 정제 {r0['정제']} · 시험구간 "
                 f"{pred.index.min():%Y-%m-%d} ~ {pred.index.max():%Y-%m-%d}", fontsize=15, color=INK, y=0.985)
    caught = ev & top_p
    caught_g = ev & top_g
    log.info("[%s|%s·%s·%s] 경보 %d시간씩 — 빙결 %d 중 PINN %d / GBM %d", train, arch, target, r0["정제"], k,
             int(ev.sum()), int(caught.sum()), int(caught_g.sum()))
    fig.text(0.05, 0.952,
             f"시드 {int(r0['시드'])} 모델 · 시험 {len(pred):,}시간 중 빙결 {int(ev.sum())}시간 ({ev.mean():.2%})   ·   "
             f"상위 1% 경보(각 {k}시간)가 잡은 빙결: PINN {int(caught.sum())} / GBM {int(caught_g.sum())}시간   ·   "
             f"AUC PINN {r0['PINN_전체_AUC']:.3f} / GBM {r0['GBM_전체_AUC']:.3f}   ·   "
             f"시각 t 의 값 = 't 부터 6시간 안에 T61 < 0℃' 에 대한 예측", fontsize=9.5, color=INK2)

    ax1 = fig.add_subplot(gs[0, :]); _axes(ax1)
    _temp(ax1, w, "① 공급온도 — 실제 vs 예측 (시험구간 전체)")
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1); _axes(ax2)
    _prob(ax2, w, thr, "② 빙결 확률 — PINN vs GBM")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ③ 월별
    ax3 = fig.add_subplot(gs[2, :]); _axes(ax3, "y")
    mon = pd.DataFrame({"빙결": ev, "PINN": caught, "GBM": caught_g}).resample("MS").sum()
    x = np.arange(len(mon))
    ax3.bar(x - 0.27, mon["빙결"], width=0.26, color=AXIS, label="빙결 시간")
    ax3.bar(x, mon["PINN"], width=0.26, color=BLUE, label="PINN 상위 1% 경보가 잡음")
    ax3.bar(x + 0.27, mon["GBM"], width=0.26, color=ORANGE, label="GBM 상위 1% 경보가 잡음")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{d:%y-%m}" for d in mon.index], fontsize=7, color=INK2, rotation=90)
    ax3.set_xlim(-0.6, len(mon) - 0.4)
    ax3.set_ylabel("시간 수", fontsize=8, color=INK2)
    ax3.set_title("③ 월별 — 빙결 시간과 경보 적중", fontsize=10.5, color=INK, pad=20)
    _legend(ax3)

    # ④⑤ 빙결이 가장 많았던 10일 확대
    if ev.any():
        roll = ev.astype(float).reindex(full, fill_value=0).rolling("10D").sum()
        end = roll.idxmax()
        z = w.loc[end - pd.Timedelta(days=10): end]
        ax4 = fig.add_subplot(gs[3, 0]); _axes(ax4)
        _temp(ax4, z, f"④ 확대 — 빙결이 가장 많았던 10일 ({z.index[0]:%Y-%m-%d}~)")
        ax5 = fig.add_subplot(gs[3, 1]); _axes(ax5)
        _prob(ax5, z, thr, "⑤ 같은 10일의 빙결 확률")
        for ax in (ax4, ax5):
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    out = out_dir / f"timeline_{train}.png"
    fig.savefig(out, dpi=100, facecolor=SURF)
    plt.close(fig)
    log.info("저장: %s", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trains", nargs="+", default=list(config.TRAINS))
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--arch", default="hard", choices=config.ARCHS)
    ap.add_argument("--target", default="min_raw", choices=config.TARGETS)
    ap.add_argument("--all", action="store_true", help="OPTUNA_DIR 아래 pred 가 있는 모든 조합 폴더")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    style.apply()
    if args.all:
        for p in sorted(config.OPTUNA_DIR.rglob("pred_*.parquet")):
            top = p.relative_to(config.OPTUNA_DIR).parts[0]
            if top == "smoke" or top.startswith("_") or not p.parent.name.startswith("v_"):
                continue
            t = p.stem.split("_")[-1]
            if (p.parent / f"metrics_seeds_{t}.csv").exists():
                plot(p.parent, t)
        return
    out_dir = config.variant_dir(args.variant, args.smoke, args.arch, args.target)
    for t in args.trains:
        plot(out_dir, t)


if __name__ == "__main__":
    main()
