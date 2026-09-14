"""열화 검정 시각화 — 추세·회복·톱니를 한 장에."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import style
from . import config, degrade as dg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    style.apply()
    daily = pd.read_csv(config.OUTPUT_DIR / "degrade_daily.csv", parse_dates=["t0"])
    tr = pd.read_csv(config.OUTPUT_DIR / "degrade_trend.csv")
    rc = pd.read_csv(config.OUTPUT_DIR / "degrade_recovery.csv")
    sw = pd.read_csv(config.OUTPUT_DIR / "degrade_sawtooth.csv")
    mt = [pd.Timestamp(d) for d in dg.MAINTENANCE]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.3,
                          left=0.06, right=0.985, top=0.87, bottom=0.06)
    fig.suptitle("열화 추적 — U·A = Q_gas/ΔT_lm (유량은 수조 열수지, 문서 12)",
                 fontsize=14, y=0.965)
    t = tr.set_index("히터")
    s = sw.set_index("히터")
    fig.text(0.06, 0.925,
             "결론: 부하·계절을 뺀 뒤 A 만 대조군을 넘는 하락 추세를 보인다"
             f"(A {t.loc['A','잔차 연변화%']:+.2f}%/년, 대조군 p95 "
             f"{t.loc['A','대조군 |연변화|p95%']:.2f}%). 그러나 **정비 시점에 초기화되지 않는다**"
             f"(톱니 검정 p {s.loc['A','p(단측)']:.2f}).\n"
             "→ 센서 드리프트와 구별되지 않는다. `TI33x`·`TI-D2x` 는 계기 교정 프로그램에 "
             "없어(교정 기록은 알람/스위치 태그뿐) 무보정 누적 드리프트가 가능하다.",
             fontsize=10, va="top", color="#333",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fdf0ee", ec="#d9b9b9"))

    for i, u in enumerate(config.UNITS):
        g = daily[daily["히터"] == u].sort_values("t0")
        if not len(g):
            continue
        # ① 원 시계열
        ax = fig.add_subplot(gs[0, i])
        ax.plot(g["t0"], g["ua_norm"] / 1e3, lw=0.4, color="#1a5a9e", alpha=0.6)
        ax.plot(g["t0"], g["ua_norm"].rolling(60, min_periods=20).median() / 1e3,
                lw=1.4, color="#c0392b")
        for d in mt:
            ax.axvline(d, color="#2e6b45", lw=0.8, ls="--", alpha=0.7)
        ax.set_title(f"① {u} — U·A (일별 중앙)", fontsize=10)
        ax.set_ylabel("kW/K" if i == 0 else "")

        # ② 부하 의존 (보정 전후)
        ax = fig.add_subplot(gs[1, i])
        X, yrs = dg._design(g.reset_index(drop=True), g["t0"].min())
        y = np.log(g["ua_norm"].to_numpy())
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        ax.scatter(g["x"], y - y.mean(), s=3, alpha=0.12, color="#7f7f7f", label="보정 전")
        ax.scatter(g["x"], r, s=3, alpha=0.18, color="#1a5a9e", label="부하·계절 제거")
        ax.axhline(0, color="#666", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("유량 / 설계")
        ax.set_title(f"② {u} — 부하 의존 (R²={t.loc[u,'부하계절_설명력R²']:.2f})", fontsize=10)
        if i == 0:
            ax.set_ylabel("log U·A 편차")
            leg = ax.legend(fontsize=7.5, markerscale=4)
            for h in leg.legend_handles:
                h.set_alpha(1.0)

        # ③ 잔차 + 추세 + 정비선
        ax = fig.add_subplot(gs[2, i])
        ax.plot(g["t0"], r, lw=0.3, color="#888", alpha=0.5)
        ax.plot(g["t0"], pd.Series(r).rolling(90, min_periods=30).median(),
                lw=1.3, color="#1a5a9e", label="90일 중앙")
        c = np.polyfit(yrs, r, 1)
        ax.plot(g["t0"], np.polyval(c, yrs), lw=1.6, color="#c0392b",
                label=f"추세 {100*(np.exp(c[0])-1):+.2f}%/년")
        for d in mt:
            ax.axvline(d, color="#2e6b45", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0, color="#666", lw=0.8)
        ax.set_title(f"③ {u} — 잔차와 추세 (초록 = 정비)", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower left")

    out = config.OUTPUT_DIR / "degrade.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log.info("저장: %s", out)


if __name__ == "__main__":
    main()
