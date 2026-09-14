"""정제 비교 그림 — 넣기 전 / 넣은 후(각각) / 세 가지 모두. 시드 반복.

형태: 조합을 **x축 범주**로 두고 한 계열색으로 그린다(5개 범주색은 팔레트 전쌍 검증을
통과하지 못하므로 쓰지 않는다 — 식별은 축 라벨이 한다). 점 = 시드별 값, 굵은 표식 = 평균,
세로막대 = ±1 표준편차. GBM 은 두 번째 색(검증 통과 쌍)의 평균 표식으로 참고만 한다.
짝비교 패널은 같은 시드끼리 뺀 `조합 − 없음` 차이와 0 기준선이다. 지표마다 따로 축(이중축 없음).
표 원자료: output/refine_seeds_summary.csv · refine_seeds_pairs.csv (그림의 테이블 뷰).
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import style
from . import config
from .refine_experiment import VARIANTS

log = logging.getLogger(__name__)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
ORDER = [v[0] for v in VARIANTS]
PANELS = [("PINN_전체_AUC", "GBM_전체_AUC", "AUC"),
          ("PINN_전체_리프트1", "GBM_전체_리프트1", "리프트 1% (배)"),
          ("PINN_전체_Brier개선%", None, "Brier 개선 (%)")]


def _axes(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    style.apply()
    raw = pd.read_csv(config.OUTPUT_DIR / "refine_seeds_raw.csv")
    n_seed = raw["시드"].nunique()
    trains = list(config.TRAINS)

    fig = plt.figure(figsize=(18, 4.6 * len(trains) + 1.0), facecolor=SURF)
    gs = fig.add_gridspec(len(trains), len(PANELS) + 1, hspace=0.55, wspace=0.28,
                          left=0.05, right=0.99, top=0.88, bottom=0.08)
    fig.suptitle(f"정제 비교 — 넣기 전 / 넣은 후(각각) / 세 가지 모두  ·  제한 PINN, 시드 {n_seed}개",
                 fontsize=14, color=INK, y=0.975)
    fig.text(0.05, 0.925, "점 = 시드별 값 · 굵은 표식 = 평균 · 막대 = ±1 표준편차 · 주황 ◆ = 같은 입력 GBM 평균   "
             "|   마지막 열: 같은 시드끼리 뺀 (조합 − 없음), 0 위면 개선", fontsize=9.5, color=INK2)

    x = np.arange(len(ORDER))
    for r, t in enumerate(trains):
        d = raw[raw["계열"] == t]
        for c, (pm, gm, label) in enumerate(PANELS):
            ax = fig.add_subplot(gs[r, c]); _axes(ax)
            for i, v in enumerate(ORDER):
                vals = d.loc[d["정제"] == v, pm].to_numpy(float)
                jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.06
                ax.scatter(i + jitter, vals, s=18, color=BLUE, alpha=0.35, lw=0, zorder=2)
                ax.errorbar(i, np.nanmean(vals), yerr=np.nanstd(vals, ddof=1) if len(vals) > 1 else 0,
                            fmt="o", ms=8, color=BLUE, ecolor=BLUE, elinewidth=2, capsize=0, zorder=3,
                            label="PINN" if i == 0 else None)
                if gm:
                    ax.scatter(i + 0.28, d.loc[d["정제"] == v, gm].mean(), marker="D", s=30,
                               color=ORANGE, zorder=3, label="GBM (평균)" if i == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels(ORDER, fontsize=8, color=INK2, rotation=15)
            ax.set_title(f"{t} — {label}", fontsize=10, color=INK)
            if c == 0:
                ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left",
                          bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
                ax.set_title(f"{t} — {label}", fontsize=10, color=INK, pad=18)

        # 짝비교: 같은 시드끼리 (조합 − 없음)
        ax = fig.add_subplot(gs[r, len(PANELS)]); _axes(ax)
        base = d[d["정제"] == ORDER[0]].set_index("시드")["PINN_전체_AUC"]
        for i, v in enumerate(ORDER[1:], start=1):
            cur = d[d["정제"] == v].set_index("시드")["PINN_전체_AUC"]
            diff = (cur - base).dropna().to_numpy()
            jitter = (np.arange(len(diff)) - (len(diff) - 1) / 2) * 0.06
            ax.scatter(i + jitter, diff, s=18, color=BLUE, alpha=0.35, lw=0, zorder=2)
            ax.errorbar(i, diff.mean(), yerr=diff.std(ddof=1) if len(diff) > 1 else 0, fmt="o", ms=8,
                        color=BLUE, elinewidth=2, capsize=0, zorder=3)
            better = int((diff > 0).sum())
            ax.annotate(f"{better}/{len(diff)} 개선", (i, diff.mean()), xytext=(8, 0),
                        textcoords="offset points", fontsize=7.5, color=INK2, va="center")
        ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
        ax.set_xticks(x[1:])
        ax.set_xticklabels(ORDER[1:], fontsize=8, color=INK2, rotation=15)
        ax.set_xlim(0.4, len(ORDER) - 0.4)
        ax.set_title(f"{t} — AUC 차이 (조합 − 없음)", fontsize=10, color=INK)

    out = config.OUTPUT_DIR / "refine_compare.png"
    fig.savefig(out, dpi=115, facecolor=SURF)
    plt.close(fig)
    log.info("저장: %s", out)


if __name__ == "__main__":
    main()
