"""진단 그림 — 조합·계열당 한 장: 학습이 제대로 됐나, 무엇을 골랐나, 얼마나 맞히나.

패널 (지표 하나·축 하나, 이중축 없음):
  ① 학습곡선 — 최종 학습의 학습/검증 손실 (시드별, 로그축)
  ② 물리 모수 수렴 — epoch 별 μ_JT (시드별) + KOGAS 0.56
  ③ Optuna 탐색 이력 — trial 별 CV AUC 와 누적 최고
  ④ fold 별 검증 AUC — 최적 하이퍼파라미터로 다시 돌린 5-fold
  ⑤ ROC · ⑥ PR — 시험구간, PINN vs 같은 입력 GBM
  ⑦ 보정 곡선 — 예측 확률 vs 실제 빙결 비율
  ⑧ 공급온도 예측 — 시험구간에서 가장 추웠던 3주: 실제 6h 내 최저 T61 vs 예측 평균 ±1σ

색: 단일 계열 blue `#2a78d6`, 두 범주(PINN/GBM, 검증/학습)는 검증 통과 쌍 blue/orange `#eb6834`.
시드 반복은 같은 blue 의 옅은 선(식별이 아니라 흩어짐을 보이려는 것). 글자는 잉크 토큰.
표 원자료: history_*·trials_*·fold_params_*·pred_*·metrics_seeds_* csv/parquet.
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from eda import style
from . import config

log = logging.getLogger(__name__)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
LEG = dict(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
           ncol=3, borderaxespad=0.2)


def _axes(ax, grid_axis="both"):
    ax.set_facecolor(SURF)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _title(ax, text, legend=False):
    ax.set_title(text, fontsize=10, color=INK, pad=18 if legend else 6)
    if legend:
        ax.legend(**LEG)


def plot(out_dir, train: str):
    hist = pd.read_csv(out_dir / f"history_{train}.csv")
    trials = pd.read_csv(out_dir / f"trials_{train}.csv")
    fp = pd.read_csv(out_dir / f"fold_params_{train}.csv")
    pred = pd.read_parquet(out_dir / f"pred_{train}.parquet")
    ms = pd.read_csv(out_dir / f"metrics_seeds_{train}.csv")
    seeds = list(ms["시드"])
    s0 = seeds[0]
    arch = ms["구조"].iloc[0] if "구조" in ms else "hard"
    target = ms["타깃"].iloc[0] if "타깃" in ms else "min_raw"
    is_nn = arch in ("nn", "lstm")

    fig = plt.figure(figsize=(22, 10.8), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.28, left=0.045, right=0.99, top=0.86, bottom=0.07)
    fig.suptitle(f"PINN 진단 [{arch}·{target}] — 계열 {train}  ·  정제 {ms['정제'].iloc[0]}  ·  해상도 {config.RES_MIN}분  ·  "
                 f"시험 {ms['시험시작'].iloc[0]}~", fontsize=14, color=INK, y=0.975)
    fig.text(0.045, 0.915,
             f"시험 AUC PINN {ms['PINN_전체_AUC'].mean():.3f} ± {ms['PINN_전체_AUC'].std(ddof=1):.3f} (시드 {len(ms)}) / "
             f"GBM {ms['GBM_전체_AUC'].iloc[0]:.3f}   ·   리프트1% {ms['PINN_전체_리프트1'].mean():.1f} / "
             f"{ms['GBM_전체_리프트1'].iloc[0]:.1f}   ·   Brier 개선 {ms['PINN_전체_Brier개선%'].mean():.1f}%   ·   "
             f"CV AUC {ms['CV_AUC'].iloc[0]:.3f}   ·   ⑤~⑧ 은 시드 {s0} 모델", fontsize=9.5, color=INK2)

    # ① 학습곡선
    ax = fig.add_subplot(gs[0, 0]); _axes(ax)
    for s in seeds[1:]:
        h = hist[hist["시드"] == s]
        ax.plot(h["epoch"], h["valid"], color=BLUE, lw=0.8, alpha=0.3)
    h0 = hist[hist["시드"] == s0]
    ax.plot(h0["epoch"], h0["train"], color=ORANGE, lw=1.4, label=f"학습 (시드 {s0})")
    ax.plot(h0["epoch"], h0["valid"], color=BLUE, lw=1.8, label=f"검증 (시드 {s0})")
    ax.plot([], [], color=BLUE, lw=0.8, alpha=0.3, label="검증 (다른 시드)")
    b = h0.loc[h0["valid"].idxmin()]
    ax.scatter([b["epoch"]], [b["valid"]], s=36, color=BLUE, edgecolor=SURF, lw=1.5, zorder=4)
    ax.annotate(f"최저 {b['valid']:.2f} @ {int(b['epoch'])}", (b["epoch"], b["valid"]), xytext=(-6, 10),
                textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("epoch", fontsize=8, color=INK2)
    ax.set_ylabel("손실 (NLL + 물리잔차, 로그축)", fontsize=8, color=INK2)
    _title(ax, "① 학습곡선 — 최종 학습", legend=True)

    # ② 물리 모수 수렴
    ax = fig.add_subplot(gs[0, 1]); _axes(ax)
    for s in seeds[1:]:
        h = hist[hist["시드"] == s]
        ax.plot(h["epoch"], h["μ_JT"], color=BLUE, lw=0.8, alpha=0.3)
    ax.plot(h0["epoch"], h0["μ_JT"], color=BLUE, lw=1.8, label=f"μ_JT (시드 {s0})")
    ax.axhline(0.56, color=MUTED, lw=1, label="KOGAS 문헌값 0.56")
    last = hist.groupby("시드").tail(1)
    ax.text(0.98, 0.04, f"최종 U·A {last['UA_kW/K'].min():.2f}~{last['UA_kW/K'].max():.2f} kW/K\n"
                        f"최종 N0 {last['N₀'].min():.3f}~{last['N₀'].max():.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xlabel("epoch", fontsize=8, color=INK2)
    ax.set_ylabel("μ_JT [℃/bar]", fontsize=8, color=INK2)
    _title(ax, "② 물리 모수 수렴 — 줄-톰슨 계수", legend=True)
    if is_nn:
        ax.text(0.5, 0.5, "물리층 없음 (nn 대조군)", transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=MUTED, bbox={"boxstyle": "round,pad=0.4", "fc": SURF, "ec": GRID})

    # ③ Optuna 탐색 이력
    ax = fig.add_subplot(gs[0, 2]); _axes(ax)
    done = trials[trials["state"] == "COMPLETE"]
    pruned = trials[trials["state"] == "PRUNED"]
    ax.scatter(done["number"], done["value"], s=26, color=BLUE, zorder=3, label="완료 trial")
    ax.step(done["number"], done["value"].cummax(), where="post", color=ORANGE, lw=1.6, label="누적 최고")
    for n in pruned["number"]:
        ax.axvline(n, color=GRID, lw=3, zorder=1)
    ax.plot([], [], color=GRID, lw=3, label=f"가지치기 {len(pruned)}개")
    ax.set_xlabel("trial", fontsize=8, color=INK2)
    ax.set_ylabel("CV AUC (5-fold 평균)", fontsize=8, color=INK2)
    _title(ax, "③ Optuna 탐색 이력", legend=True)

    # ④ fold 별 검증 AUC
    ax = fig.add_subplot(gs[0, 3]); _axes(ax, "y")
    k = fp["fold"].str.replace("fold", "").astype(int)
    ax.bar(k, fp["검증AUC"], color=BLUE, width=0.6)
    for xi, a, e in zip(k, fp["검증AUC"], fp["검증사건"]):
        ax.text(xi, a, f"{a:.3f}\n사건 {e}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.axhline(0.5, color=AXIS, lw=1)
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(k)
    ax.set_xlabel("fold (1 = 가장 이른 검증창)", fontsize=8, color=INK2)
    ax.set_ylabel("검증 AUC", fontsize=8, color=INK2)
    _title(ax, "④ fold 별 검증 AUC — 최적 하이퍼파라미터")

    ev = pred["ev"].to_numpy().astype(int)
    # ⑤ ROC
    ax = fig.add_subplot(gs[1, 0]); _axes(ax)
    for col, c, name in (("p", BLUE, "PINN"), ("p_gbm", ORANGE, "GBM")):
        fpr, tpr, _ = roc_curve(ev, pred[col])
        ax.plot(fpr, tpr, color=c, lw=1.8, label=f"{name} AUC {roc_auc_score(ev, pred[col]):.3f}")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set_xlabel("오경보율 (FPR)", fontsize=8, color=INK2)
    ax.set_ylabel("탐지율 (TPR)", fontsize=8, color=INK2)
    _title(ax, "⑤ ROC — 시험구간", legend=True)

    # ⑥ PR
    ax = fig.add_subplot(gs[1, 1]); _axes(ax)
    for col, c, name in (("p", BLUE, "PINN"), ("p_gbm", ORANGE, "GBM")):
        pr, rc, _ = precision_recall_curve(ev, pred[col])
        ax.plot(rc, pr, color=c, lw=1.8, label=f"{name} AP {average_precision_score(ev, pred[col]):.3f}")
    ax.axhline(ev.mean(), color=AXIS, lw=1, label=f"사건률 {ev.mean():.3%}")
    ax.set_xlabel("재현율", fontsize=8, color=INK2)
    ax.set_ylabel("정밀도", fontsize=8, color=INK2)
    _title(ax, "⑥ PR — 시험구간", legend=True)

    # ⑦ 보정 곡선 (예측 분위 구간)
    ax = fig.add_subplot(gs[1, 2]); _axes(ax)
    top = 0.0
    for col, c, name in (("p", BLUE, "PINN"), ("p_gbm", ORANGE, "GBM")):
        p = pred[col].to_numpy()
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        b_ = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        g = pd.DataFrame({"p": p, "ev": ev, "b": b_}).groupby("b").agg(pm=("p", "mean"), om=("ev", "mean"), n=("ev", "size"))
        ax.plot(g["pm"], g["om"], color=c, lw=1.6, marker="o", ms=5, label=name)
        top = max(top, g["pm"].max(), g["om"].max())
    top = max(top * 1.08, 0.02)
    ax.plot([0, top], [0, top], color=AXIS, lw=1, label="완전 보정")
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel("예측 확률 (구간 평균)", fontsize=8, color=INK2)
    ax.set_ylabel("실제 빙결 비율", fontsize=8, color=INK2)
    _title(ax, "⑦ 보정 곡선 — 예측 분위 10구간", legend=True)

    # ⑧ 공급온도 예측 — 가장 추웠던 3주
    ax = fig.add_subplot(gs[1, 3]); _axes(ax)
    if "y" in pred:
        t_min = pred["y"].idxmin()
        w = pred.loc[t_min - pd.Timedelta(days=10): t_min + pd.Timedelta(days=11)]
        ax.fill_between(w.index, w["mu"] - w["sd"], w["mu"] + w["sd"], color=BLUE, alpha=0.15, lw=0, label="예측 ±1σ")
        ax.plot(w.index, w["y"], color=INK2, lw=0.9, label="실제 (6h 내 1시간평균 최저)" if target == "mean" else "실제 (6h 내 분 단위 최저)")
        ax.plot(w.index, w["mu"], color=BLUE, lw=1.4, label="예측 평균")
        e = w[w["ev"] > 0.5]
        ax.scatter(e.index, e["y"], s=10, color=ORANGE, zorder=4, label="빙결 사건")
        ax.axhline(config.FREEZE_C, color=AXIS, lw=1)
        rmse = float(np.sqrt(np.mean((pred["mu"] - pred["y"]) ** 2)))
        ax.text(0.98, 0.04, f"시험 전체 RMSE {rmse:.2f}℃", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
        ax.tick_params(axis="x", labelrotation=25)
    else:
        ax.text(0.5, 0.5, "pred 에 실제값(y) 없음 — 이전 판 산출물", ha="center", va="center", color=MUTED)
    ax.set_ylabel("공급온도 T61 [℃]", fontsize=8, color=INK2)
    ax.set_title("⑧ 공급온도 예측 — 시험구간 최저온 전후 3주", fontsize=10, color=INK, pad=30)
    if "y" in pred:                                  # 범례 4개 → 2줄로 올려 제목과 겹치지 않게
        ax.legend(**{**LEG, "ncol": 2})

    out = out_dir / f"diag_{train}.png"
    fig.savefig(out, dpi=110, facecolor=SURF)
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
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    style.apply()
    out_dir = config.variant_dir(args.variant, args.smoke, args.arch, args.target)
    for t in args.trains:
        plot(out_dir, t)


if __name__ == "__main__":
    main()
