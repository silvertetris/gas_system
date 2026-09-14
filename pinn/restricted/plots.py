"""제한 PINN 결과 그림 — "식이 성립하는 곳에서 예측이 되는가"를 한 장에.

색: 계열 1 blue `#2a78d6` = PINN(모델), 계열 2 orange `#eb6834` = GBM/경험값.
두 색 쌍은 dataviz 검증기 통과(light). 글자는 잉크 토큰, 격자는 실선 헤어라인.
실제 빙결은 상태색 critical `#d03b3b` + 라벨로만 쓴다.
표 형태 원자료: output/metrics.csv · flowcheck.csv (그림의 테이블 뷰).
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import style
from . import config

log = logging.getLogger(__name__)

BLUE, ORANGE, CRIT = "#2a78d6", "#eb6834", "#d03b3b"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _axes(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    style.apply()
    m = pd.read_csv(config.OUTPUT_DIR / "metrics.csv").set_index("계열")
    fc = pd.read_csv(config.OUTPUT_DIR / "flowcheck.csv")

    fig = plt.figure(figsize=(18, 9.6), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.28,
                          left=0.045, right=0.99, top=0.84, bottom=0.08)
    fig.suptitle("제한 PINN — 성립이 확인된 열전달식 (Q5 줄-톰슨 + Q6 매설배관) 만 사용",
                 fontsize=14, color=INK, y=0.975)
    txt = "   |   ".join(
        f"{t}: μ_JT {m.loc[t,'μ_JT']:.3f} ℃/bar · U·A_배관 {m.loc[t,'UA_배관_kW/K']:.2f} kW/K · "
        f"N0 {m.loc[t,'N₀']:.3f} · T_g {m.loc[t,'T_g']}" for t in m.index)
    fig.text(0.045, 0.915, "식별된 물리 모수 — " + txt, fontsize=9.5, color=INK2)

    for r, t in enumerate(m.index):
        pred = pd.read_parquet(config.OUTPUT_DIR / f"pred_{t}.parquet")

        # ① 유량 구간별 헤더보존율 — 식이 성립하는 곳
        ax = fig.add_subplot(gs[r, 0]); _axes(ax)
        c = fc[fc["계열"] == t]
        x = np.arange(len(c))
        ax.plot(x, c["k_모델 exp(−N)"], color=BLUE, lw=2, marker="o", ms=5, label="모델 exp(−N)")
        ax.plot(x, c["k_경험(회귀)"], color=ORANGE, lw=2, marker="o", ms=5, label="경험 (구간 회귀)")
        ax.set_xticks(x)
        ax.set_xticklabels(c["유량구간"], rotation=25, fontsize=7, color=MUTED)
        ax.set_xlabel("계열 총유량 구간 [kg/s]", fontsize=8, color=INK2)
        ax.set_ylabel("헤더신호 보존율 k", fontsize=8, color=INK2)
        ax.set_title(f"① {t} — 유량이 크면 헤더 온도가 더 남아야 한다", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2)

        # ② 빙결 예측 AUC — 부분집합별 (한 축, 한 지표)
        ax = fig.add_subplot(gs[r, 1]); _axes(ax)
        tags = ["전체", "유량관측", "유량미관측"]
        w = 0.36
        for j, (name, col) in enumerate((("PINN", BLUE), ("GBM", ORANGE))):
            vals = [m.loc[t, f"{name}_{g}_AUC"] for g in tags]
            xs = np.arange(len(tags)) + (j - 0.5) * (w + 0.02)
            ax.bar(xs, vals, width=w, color=col, label=name)
            for xi, v in zip(xs, vals):
                if np.isfinite(v):
                    ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(np.arange(len(tags)))
        ax.set_xticklabels(tags, fontsize=8, color=INK2)
        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel("ROC AUC (시험 2022~)", fontsize=8, color=INK2)
        ax.set_title(f"② {t} — 빙결 예측 AUC", fontsize=10, pad=18)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left",
                  bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)

        # ③ 동시점 물리 정합 — 유량 관측된 시각만
        ax = fig.add_subplot(gs[r, 2]); _axes(ax)
        d = pred[pred["m_obs"]]
        if len(d) > 20000:
            d = d.sample(20000, random_state=0)
        ax.scatter(d["t61_now"], d["phys_now"], s=3, color=BLUE, alpha=0.18, lw=0)
        lo = float(np.nanpercentile(d[["t61_now", "phys_now"]], 0.5))
        hi = float(np.nanpercentile(d[["t61_now", "phys_now"]], 99.5))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("실측 공급온도 T_61 [℃]", fontsize=8, color=INK2)
        ax.set_ylabel("물리층 계산 [℃]", fontsize=8, color=INK2)
        ax.set_title(f"③ {t} — 식만으로 공급온도 재현 (RMSE {m.loc[t,'물리정합RMSE_유량관측']:.2f}℃)",
                     fontsize=10)

        # ④ 시험구간 위험확률과 실제 빙결
        ax = fig.add_subplot(gs[r, 3]); _axes(ax)
        ax.plot(pred.index, pred["p"], color=BLUE, lw=1.0, label="P(빙결) — PINN")
        evt = pred.index[pred["ev"] > 0.5]
        ax.scatter(evt, np.full(len(evt), 1.03), s=10, color=CRIT, marker="v", lw=0,
                   label="실제 빙결 (T_61<0℃)", clip_on=False)
        ax.set_ylim(0, 1.06)
        ax.set_ylabel("확률", fontsize=8, color=INK2)
        ax.set_title(f"④ {t} — 시험구간 위험확률 (사건률 {100*m.loc[t,'시험사건률']:.2f}%)",
                     fontsize=10, pad=18)
        # 확률선이 축 전체를 가로질러 범례를 둘 빈 공간이 없다 → ②와 같이 축 위쪽 밖으로
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left",
                  bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
        for lab in ax.get_xticklabels():
            lab.set_rotation(25); lab.set_fontsize(7)

    out = config.OUTPUT_DIR / "restricted_pinn.png"
    fig.savefig(out, dpi=115, facecolor=SURF)
    plt.close(fig)
    log.info("저장: %s", out)


if __name__ == "__main__":
    main()
