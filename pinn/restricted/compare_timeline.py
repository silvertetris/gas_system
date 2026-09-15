"""모델별 빙결 예측 시계열 비교 (그림) + 정확도 표 (따로).

그림 — 행 = 모델(GBM · hard · soft · softw · nn · LSTM), 열 3개, 시간축 공유:
  ① 시험구간 전체 빙결 확률 + 실제 빙결 시각(아래 눈금) + 모델별 상위 1% 경보선
  ② 빙결이 가장 많았던 10일 — 빙결 확률
  ③ 같은 10일 — 공급온도 예측 평균 ±1σ vs 실제 (GBM 은 온도 예측이 없다)
작은 배수(small multiples) 형식이라 모델은 행 이름으로 구분하고 선은 한 색(blue)만 쓴다. 실제값·사건은 잉크색.
**정확도 숫자는 그림에 넣지 않는다** — 표(csv·md)로 분리.

표 — 시드 5 평균 ± 표준편차(AUC·리프트·Brier) + 시드 42 모델(AP·경보 적중·정밀도·재현율·공급온도 RMSE).
경보는 모델마다 **같은 시간 수(시험구간 상위 1%)** 로 맞춘다.

    PYTHONPATH=. .venv/bin/python -m pinn.restricted.compare_timeline --variants KF 없음 --target min
"""
from __future__ import annotations

import argparse
import logging
import unicodedata

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from eda import style
from . import config

log = logging.getLogger(__name__)
BLUE = "#2a78d6"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
MODELS = [("gbm", "GBM (같은 입력)"), ("hard", "hard PINN\n(출력 = 물리식)"), ("soft", "soft PINN\n(물리 = 손실)"),
          ("softw", "soft PINN\n(물리 가중 확장)"), ("nn", "nn\n(물리 없음)"), ("lstm", "LSTM\n(시계열, 물리 없음)")]
TNAME = {"min": "6h 내 분 단위 최저 T61 < 0℃ (센서 이상값 제거)", "mean": "6h 내 1시간 평균 최저 T61 < 0℃"}


def _dir(arch: str, target: str, variant: str):
    base = config.variant_dir("auto", False, arch, target)
    if not base.exists():
        return None
    want = unicodedata.normalize("NFC", f"v_{variant}")
    return next((p for p in base.iterdir() if unicodedata.normalize("NFC", p.name) == want), None)


def _axes(ax, grid="y"):
    ax.set_facecolor(SURF)
    ax.grid(True, axis=grid, color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7.5)


def load(train: str, target: str, variant: str):
    """모델 → (pred DataFrame [p, mu, sd], metrics_seeds DataFrame 또는 None). 실행 중이라 없으면 건너뛴다."""
    out, ref = {}, None
    for arch, _ in MODELS[1:]:
        d = _dir(arch, target, variant)
        if d is None or not (d / f"pred_{train}.parquet").exists() or not (d / f"metrics_seeds_{train}.csv").exists():
            continue
        pred = pd.read_parquet(d / f"pred_{train}.parquet").sort_index()
        out[arch] = (pred[["p", "mu", "sd"]], pd.read_csv(d / f"metrics_seeds_{train}.csv"))
        if ref is None:
            ref = pred
    if ref is None:
        return None, {}
    out = {"gbm": (ref[["p_gbm"]].rename(columns={"p_gbm": "p"}).assign(mu=np.nan, sd=np.nan), None), **out}
    return ref[["y", "ev"]], out


def table(train: str, target: str, variant: str, truth: pd.DataFrame, models: dict) -> pd.DataFrame:
    ev = (truth["ev"] > 0.5).to_numpy()
    k = int(np.ceil(0.01 * len(truth)))
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for arch, name in MODELS:
        if arch not in models:
            rows.append({"모델": name.replace("\n", " "), "상태": "실행 중 — 결과 없음"})
            continue
        pred, ms = models[arch]
        p = pred["p"].reindex(truth.index).to_numpy()
        tie = pred["mu"].reindex(truth.index).to_numpy() if arch != "gbm" else rng.random(len(p))
        top = np.zeros(len(p), bool)
        top[np.lexsort((np.nan_to_num(tie, nan=0.0), -p))[:k]] = True
        hit = int((top & ev).sum())
        r = {"모델": name.replace("\n", " ")}
        if ms is not None:
            r.update({"시험 AUC": f"{ms['PINN_전체_AUC'].mean():.3f} ± {ms['PINN_전체_AUC'].std(ddof=1):.3f}",
                      "리프트 1%": f"{ms['PINN_전체_리프트1'].mean():.1f}", "리프트 5%": f"{ms['PINN_전체_리프트5'].mean():.1f}",
                      "Brier 개선 %": f"{ms['PINN_전체_Brier개선%'].mean():.1f}", "CV AUC": f"{ms['CV_AUC'].iloc[0]:.3f}"})
        else:
            g = models["soft"][1] if "soft" in models else next(m for _, m in models.values() if m is not None)
            r.update({"시험 AUC": f"{g['GBM_전체_AUC'].iloc[0]:.3f}", "리프트 1%": f"{g['GBM_전체_리프트1'].iloc[0]:.1f}",
                      "리프트 5%": f"{g['GBM_전체_리프트5'].iloc[0]:.1f}", "Brier 개선 %": f"{g['GBM_전체_Brier개선%'].iloc[0]:.1f}",
                      "CV AUC": "—"})
        r.update({"AP (PR 곡선 면적)": f"{average_precision_score(ev, p):.3f}",
                  "빙결 시간": int(ev.sum()), "상위 1% 경보 시간": k, "경보가 잡은 빙결": hit,
                  "경보 정밀도 %": f"{100 * hit / k:.1f}", "빙결 재현율 %": f"{100 * hit / ev.sum():.1f}",
                  "공급온도 RMSE ℃": "—" if arch == "gbm" else
                  f"{np.sqrt(np.nanmean((pred['mu'].reindex(truth.index) - truth['y']) ** 2)):.2f}"})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.insert(0, "정제", variant)
    t.insert(0, "계열", train)
    return t


def plot(train: str, target: str, variant: str, truth: pd.DataFrame, models: dict):
    full = pd.date_range(truth.index.min(), truth.index.max(), freq=config.FREQ)
    tr = truth.reindex(full)
    ev = truth["ev"] > 0.5
    roll = ev.astype(float).reindex(full, fill_value=0).rolling("10D").sum()
    end = roll.idxmax()
    z0, z1 = end - pd.Timedelta(days=10), end
    k = int(np.ceil(0.01 * len(truth)))
    shown = [(a, n) for a, n in MODELS if a in models]
    n = len(shown)
    fig = plt.figure(figsize=(22, 2.55 * n + 1.6), facecolor=SURF)
    H = 2.55 * n + 1.6
    gs = fig.add_gridspec(n, 3, width_ratios=[2.4, 1, 1], hspace=0.28, wspace=0.12,
                          left=0.085, right=0.99, top=1 - 1.25 / H, bottom=0.55 / H)
    fig.suptitle(f"모델별 빙결 예측 시계열 — 계열 {train} · 정제 {variant} · 타깃 {TNAME[target]}",
                 fontsize=14, color=INK, y=1 - 0.3 / H)
    fig.text(0.085, 1 - 0.72 / H,
             f"시험구간 {truth.index.min():%Y-%m-%d} ~ {truth.index.max():%Y-%m-%d} (학습·튜닝에 쓰지 않은 마지막 20%) · 시드 42 모델 · "
             f"회색 선 = 모델별 상위 1% 경보선({k}시간) · 검은 눈금 = 실제 빙결 · 정확도는 별도 표", fontsize=9.5, color=INK2)
    ev_t = truth.index[ev]
    ymin_t = float(np.nanmin(tr.loc[z0:z1, "y"]))
    for r, (arch, name) in enumerate(shown):
        pred = models[arch][0].reindex(full)
        p = pred["p"]
        thr = float(np.sort(models[arch][0]["p"].to_numpy())[::-1][k - 1])
        # ① 전체
        ax = fig.add_subplot(gs[r, 0]); _axes(ax)
        ax.plot(full, p, color=BLUE, lw=0.7)
        ax.axhline(thr, color=MUTED, lw=0.9)
        ax.vlines(ev_t, -0.12, -0.04, color=INK, lw=0.5)
        ax.set_ylim(-0.14, 1.05)
        ax.set_ylabel(name, fontsize=9, color=INK, rotation=0, ha="right", va="center", labelpad=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.axvspan(z0, z1, color=GRID, alpha=0.6, lw=0)
        if r == 0:
            ax.set_title("① 시험구간 전체 — P(6h 내 빙결)   (음영 = 오른쪽 확대 구간)", fontsize=10, color=INK, loc="left")
        if r < n - 1:
            ax.tick_params(labelbottom=False)
        # ② 확대 확률
        ax = fig.add_subplot(gs[r, 1]); _axes(ax)
        zz = pred.loc[z0:z1]
        ax.plot(zz.index, zz["p"], color=BLUE, lw=1.2)
        ax.axhline(thr, color=MUTED, lw=0.9)
        ax.vlines(ev_t[(ev_t >= z0) & (ev_t <= z1)], -0.12, -0.04, color=INK, lw=0.6)
        ax.set_ylim(-0.14, 1.05)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        if r == 0:
            ax.set_title(f"② 확대 {z0:%Y-%m-%d}~ — P(6h 내 빙결)", fontsize=10, color=INK, loc="left")
        if r < n - 1:
            ax.tick_params(labelbottom=False)
        # ③ 확대 온도
        ax = fig.add_subplot(gs[r, 2]); _axes(ax)
        tz = tr.loc[z0:z1]
        ax.axhline(config.FREEZE_C, color=AXIS, lw=1)
        ax.plot(tz.index, tz["y"], color=INK, lw=1.0)
        e = tz[tz["ev"] > 0.5]
        ax.scatter(e.index, e["y"], s=7, color=INK, lw=0, zorder=4)
        if arch != "gbm":
            ax.fill_between(zz.index, zz["mu"] - zz["sd"], zz["mu"] + zz["sd"], color=BLUE, alpha=0.15, lw=0)
            ax.plot(zz.index, zz["mu"], color=BLUE, lw=1.2)
        else:
            ax.text(0.98, 0.95, "GBM 은 빙결 확률만 낸다 (공급온도 예측 없음)", transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, color=INK2, bbox={"boxstyle": "round,pad=0.3", "fc": SURF, "ec": GRID})
        ax.set_ylim(min(ymin_t, -1) - 1.5, float(np.nanmax(tz["y"])) + 3)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        if r == 0:
            ax.set_title("③ 같은 구간 — 공급온도 [℃]  검정 = 실제, 파랑 = 예측 ±1σ", fontsize=10, color=INK, loc="left")
        if r < n - 1:
            ax.tick_params(labelbottom=False)
    out = config.OPTUNA_DIR / f"model_timeline_{train}_{target}_{variant}.png"
    fig.savefig(out, dpi=110, facecolor=SURF)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["KF", "없음"])
    ap.add_argument("--target", default="min", choices=["min", "mean"])
    ap.add_argument("--trains", nargs="+", default=list(config.TRAINS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    style.apply()
    tabs = []
    for v in args.variants:
        for t in args.trains:
            truth, models = load(t, args.target, v)
            if truth is None:
                log.warning("결과 없음: %s · %s", t, v)
                continue
            log.info("저장: %s", plot(t, args.target, v, truth, models))
            tabs.append(table(t, args.target, v, truth, models))
    tab = pd.concat(tabs, ignore_index=True)
    stem = config.OPTUNA_DIR / f"model_accuracy_{args.target}"
    tab.to_csv(f"{stem}.csv", index=False, encoding="utf-8-sig")
    t2 = tab.fillna("").astype(str)
    md = ["| " + " | ".join(t2.columns) + " |", "|" + "---|" * len(t2.columns)]
    md += ["| " + " | ".join(r) + " |" for r in t2.to_numpy()]
    with open(f"{stem}.md", "w", encoding="utf-8") as fh:              # tabulate 없이 직접 작성
        fh.write("\n".join(md) + "\n")
    log.info("표: %s.csv · .md", stem)
    pd.set_option("display.width", 260)
    print(tab.fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
