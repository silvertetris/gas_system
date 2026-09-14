"""XAI 그림 — 계열당 한 장.

형태: 패널마다 지표 하나·축 하나(이중축 없음). 단일 계열은 blue `#2a78d6`, 사건/평시·PINN/GBM 같은
두 범주는 검증 통과 쌍 blue/orange `#eb6834`. 글자는 잉크 토큰, 격자는 실선 헤어라인.
표 원자료: output/optuna/ 의 shap_*·perm_groups_*·physics_decomp_*·fold_params_*·param_importance_* csv.
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import style
from . import config

log = logging.getLogger(__name__)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"


def _axes(ax, grid_axis="x"):
    ax.set_facecolor(SURF)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot(out_dir, train: str):
    feat = pd.read_csv(out_dir / f"shap_features_{train}.csv")
    grp = pd.read_csv(out_dir / f"shap_groups_{train}.csv", index_col=0)
    phys = pd.read_csv(out_dir / f"physics_decomp_{train}.csv", index_col=0)
    perm = pd.read_csv(out_dir / f"perm_groups_{train}.csv")
    fp = pd.read_csv(out_dir / f"fold_params_{train}.csv")
    mp = out_dir / f"metrics_seeds_{train}.csv"
    metrics = (pd.read_csv(mp).iloc[0] if mp.exists()
               else pd.read_csv(out_dir / "metrics.csv").set_index("계열").loc[train])
    imp_path = out_dir / f"param_importance_{train}.csv"
    imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0] if imp_path.exists() else pd.Series(dtype=float)
    arch, target = metrics.get("구조", "hard"), metrics.get("타깃", "min_raw")
    is_nn = arch in ("nn", "lstm")

    fig = plt.figure(figsize=(18, 11), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34, left=0.07, right=0.99, top=0.86, bottom=0.06)
    fig.suptitle(f"PINN XAI [{arch}·{target}] — 계열 {train}  ·  정제 {metrics['정제']}  ·  시험 {metrics['시험시작']}~",
                 fontsize=14, color=INK, y=0.975)
    fig.text(0.07, 0.915,
             f"시험 AUC PINN {metrics['PINN_전체_AUC']:.3f} / GBM {metrics['GBM_전체_AUC']:.3f}   ·   "
             f"리프트1% {metrics['PINN_전체_리프트1']:.1f} / {metrics['GBM_전체_리프트1']:.1f}   ·   "
             f"CV AUC {metrics['CV_AUC']:.3f}   ·   "
             + ("물리층 없음 (nn 대조군)" if is_nn else
                f"μ_JT {metrics['μ_JT']:.3f} · U·A {metrics['UA_배관_kW/K']:.2f} kW/K · N0 {metrics['N₀']:.3f}"),
             fontsize=9.5, color=INK2)

    # ① 상위 특징 |SHAP|
    ax = fig.add_subplot(gs[0, 0]); _axes(ax)
    top = feat.head(15).iloc[::-1]
    ax.barh(top["특징"], top["전체|SHAP|"], color=BLUE, height=0.7)
    for yv, (v, g) in enumerate(zip(top["전체|SHAP|"], top["그룹"])):
        ax.text(v, yv, f"  {g}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("평균 |SHAP| (위험점수)", fontsize=8, color=INK2)
    ax.set_title("① 위험에 기여한 입력 — 상위 15", fontsize=10, color=INK)

    # ② 그룹 비중: 사건 vs 평시
    ax = fig.add_subplot(gs[0, 1]); _axes(ax)
    g = grp.sort_values("전체|SHAP|")
    y = np.arange(len(g))
    ax.barh(y + 0.19, g["사건_비중%"], height=0.36, color=ORANGE, label="빙결 사건")
    ax.barh(y - 0.19, g["평시_비중%"], height=0.36, color=BLUE, label="평시")
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=8, color=INK2)
    ax.set_xlabel("|SHAP| 비중 (%)", fontsize=8, color=INK2)
    ax.set_title("② 그룹별 기여 비중 — 사건과 평시", fontsize=10, color=INK, pad=18)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, borderaxespad=0.2)

    # ③ 그룹 순열 중요도
    ax = fig.add_subplot(gs[0, 2]); _axes(ax)
    pp = perm.sort_values("AUC하락_평균")
    ax.barh(pp["그룹"], pp["AUC하락_평균"], xerr=pp["AUC하락_표준편차"], color=BLUE, height=0.7,
            error_kw={"ecolor": INK2, "elinewidth": 1})
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel(f"AUC 하락 (기준 {perm['기준AUC(보정전)'].iloc[0]:.3f}, 5회 반복 ±1sd)", fontsize=8, color=INK2)
    ax.set_title("③ 그룹을 섞으면 AUC 가 얼마나 떨어지나 (시험구간)", fontsize=10, color=INK)

    # ④ 물리 분해 사건−평시
    ax = fig.add_subplot(gs[1, 0]); _axes(ax, "y")
    terms = ["지중", "헤더", "줄톰슨", "T61_확실성등가"]
    # 위첨자(⁻ᴺ)는 CJK 글꼴에 글리프가 없어 네모로 깨진다 → exp(−N) 로 쓴다
    labels = ["지중\nT_g·(1−exp(−N))", "헤더\nexp(−N)·T_hdr", "줄-톰슨\n−exp(−N)·μΔP", "공급온도\n합계"]
    x = np.arange(len(terms))
    ax.bar(x - 0.19, phys.loc["평시", terms], width=0.36, color=BLUE, label="평시")
    ax.bar(x + 0.19, phys.loc["사건", terms], width=0.36, color=ORANGE, label="빙결 사건")
    for xi, t in zip(x, terms):
        d = phys.loc["사건−평시", t]
        top_v = max(phys.loc["평시", t], phys.loc["사건", t], 0)
        ax.text(xi, top_v, f"Δ{d:+.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, color=INK2)
    ax.set_ylabel("℃ (6h 뒤 상태 평균)", fontsize=8, color=INK2)
    ax.set_title("④ 물리 분해 — 무엇이 공급온도를 끌어내렸나", fontsize=10, color=INK, pad=18)
    if is_nn:
        ax.text(0.5, 0.5, "물리층 없음 (nn 대조군)", transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=MUTED, bbox={"boxstyle": "round,pad=0.4", "fc": SURF, "ec": GRID})
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, borderaxespad=0.2)

    # ⑤ fold 별 물리 모수 안정성
    ax = fig.add_subplot(gs[1, 1]); _axes(ax, "y")
    folds_ = fp["fold"].str.replace("fold", "").astype(int)
    ax.plot(folds_, fp["μ_JT"], marker="o", ms=7, lw=2, color=BLUE, label="fold 별 μ_JT")
    ax.axhline(0.56, color=MUTED, lw=1, ls="-", label="KOGAS 문헌값 0.56")
    ax.set_xticks(folds_)
    ax.set_xlim(folds_.min() - 0.5, folds_.max() + 0.5)
    ax.set_xlabel("fold (1 = 가장 이른 검증창)", fontsize=8, color=INK2)
    ax.set_ylabel("μ_JT [℃/bar]", fontsize=8, color=INK2)
    # 주석(3줄)이 점 위에 올라가므로 위쪽 여백을 넉넉히 — 제목·기준선 라벨과 겹치지 않게
    lo, hi = min(fp["μ_JT"].min(), 0.56), max(fp["μ_JT"].max(), 0.56)
    span = max(hi - lo, 0.02)
    ax.set_ylim(lo - 0.12 * span, hi + 0.45 * span)
    for xi, (u, n0, a) in enumerate(zip(fp["UA_kW/K"], fp["N₀"], fp["검증AUC"])):
        ax.annotate(f"UA {u:.1f}\nN0 {n0:.2f}\nAUC {a:.2f}", (folds_.iloc[xi], fp["μ_JT"].iloc[xi]),
                    xytext=(0, 9), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=INK2,
                    bbox={"boxstyle": "round,pad=0.15", "fc": SURF, "ec": "none", "alpha": 0.9}, zorder=4)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=2, borderaxespad=0.2)
    ax.set_title("⑤ fold 별 물리 모수 — 기간이 바뀌어도 같은 물리인가", fontsize=10, color=INK, pad=18)
    if is_nn:
        ax.text(0.5, 0.5, "물리층 없음 (nn 대조군)", transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=MUTED, bbox={"boxstyle": "round,pad=0.4", "fc": SURF, "ec": GRID})

    # ⑥ Optuna 하이퍼파라미터 중요도
    ax = fig.add_subplot(gs[1, 2]); _axes(ax)
    if len(imp):
        ii = imp.sort_values()
        ax.barh(ii.index, ii.values, color=BLUE, height=0.7)
        ax.set_xlabel("fANOVA 중요도", fontsize=8, color=INK2)
    else:
        ax.text(0.5, 0.5, "완료 trial 부족 — 계산 생략", ha="center", va="center", color=MUTED)
    ax.set_title("⑥ 무엇이 CV 성능을 좌우했나" + (" (정제 조합 포함)" if "refine" in imp.index else ""),
                 fontsize=10, color=INK)

    out = out_dir / f"xai_{train}.png"
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
