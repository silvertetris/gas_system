"""PINN 파일럿 시각화 — 예측·물리분해·캘리브레이션을 한 장에.

`train.py` 가 남긴 `pred_<계열>.csv` 를 읽는다. 히터 물리(ε-NTU·합류점)가 물리층에
들어간 뒤로 `eps_<u>`(예측)·`eps_obs_<u>`(실측)·`f_<u>`(혼합가중)·`beta`·동시점 정합용
`out_obs/out_now`·`hdr_obs/hdr_now` 컬럼이 있다 — ⑦⑧ 패널이 그걸 쓴다.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import style
from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _load(train: str) -> pd.DataFrame:
    d = pd.read_csv(config.OUTPUT_DIR / f"pred_{train}.csv", index_col=0, parse_dates=True)
    m = pd.read_csv(config.OUTPUT_DIR / "metrics.csv")
    d.attrs["meta"] = m[m["계열"] == train].iloc[0].to_dict()
    return d


def plot(train: str):
    style.apply()
    d = _load(train)
    meta = d.attrs["meta"]
    ev = d["ev"].to_numpy().astype(bool)
    mu_jt = float(meta["μ_JT"])
    jt = mu_jt * d["mu_dp"].to_numpy()

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 3, hspace=0.52, wspace=0.28,
                          left=0.055, right=0.985, top=0.905, bottom=0.05)
    spec = config.TRAINS[train]
    fig.suptitle(
        f"PINN 빙결 위험 파일럿 — 계열 {train} ({'+'.join(spec['units'])})   "
        f"타깃 {spec['t61']} < 0℃,  지평 {config.HORIZON_H}h,  시험 2016년",
        fontsize=14, y=0.965)
    ua = "  ".join(f"{u} {meta[f'가중중앙_{u}']:.2f}" for u in spec["units"])
    fig.text(0.055, 0.925,
             f"학습된 물리모수:  μ_JT = {mu_jt:.4f} ℃/bar (허용 {config.MU_JT_MIN}~{config.MU_JT_MAX}, "
             f"문헌 0.4~0.5)   T_g = {meta['T_g']}   헤더보존율 exp(−N) = {meta['헤더보존율']:.3f}   "
             f"혼합가중 {ua}\n"
             f"사건률 훈련 {meta['훈련사건률']:.1%} → 시험 {meta['시험사건률']:.2%}   |   "
             f"AUC PINN {meta['PINN보정_AUC']:.3f} / GBM {meta['GBM_AUC']:.3f}   "
             f"리프트1% PINN {meta['PINN보정_리프트1']:.1f} / GBM {meta['GBM_리프트1']:.1f}",
             fontsize=9.5, color="#333", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="#eef4f8", ec="#b9ccd9"))

    # ① 위험확률 시계열 + 실제 사건
    ax = fig.add_subplot(gs[0, :])
    ax.plot(d.index, d["p"], lw=0.6, color="#1a5a9e", label="P(빙결) 예측")
    ax.fill_between(d.index, 0, ev.astype(float), color="#c0392b", alpha=0.28,
                    step="mid", lw=0, label="실제 사건")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("확률")
    ax.set_title("① 위험확률과 실제 사건 (시험구간)")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    # ② 물리 분해
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(d.index, d["mu_hdr"], lw=0.7, color="#2e6b45", label="예측 헤더온도 μ_hdr")
    ax.plot(d.index, -jt, lw=0.7, color="#9b6bbf", label=f"JT 강하 μ_JT·dp")
    ax.plot(d.index, d["mu"], lw=0.9, color="#1a5a9e", label="예측 공급온도 μ")
    ax.axhline(0, color="#c0392b", ls="--", lw=1.0, label="빙결점 0℃")
    ymin = ax.get_ylim()[0]
    ax.fill_between(d.index, ymin, ymin + (ax.get_ylim()[1] - ymin) * 0.06 * ev,
                    color="#c0392b", alpha=0.3, step="mid", lw=0)
    ax.set_ylabel("℃")
    ax.plot(d.index, d["t_g"], lw=0.7, color="#b8860b", ls=":", label="지중온도 T_g")
    ax.set_title("② 물리 분해 — T_61 = T_g + (T_hdr − μ_JT·ΔP − T_g)·exp(−N)  (하단 띠 = 실제 사건)")
    ax.legend(loc="lower left", ncol=2, fontsize=7.5, framealpha=0.9)

    # ③ 사건 vs 평시 분해 막대
    ax = fig.add_subplot(gs[1, 2])
    labels = ["헤더온도\nμ_hdr", "JT 강하\nμ_JT·dp"]
    evm = [float(d["mu_hdr"][ev].mean()), float(jt[ev].mean())]
    nom = [float(d["mu_hdr"][~ev].mean()), float(jt[~ev].mean())]
    x = np.arange(2)
    ax.bar(x - 0.2, evm, 0.38, label="사건 시", color="#c0392b", alpha=0.85)
    ax.bar(x + 0.2, nom, 0.38, label="평시", color="#4f8cd9", alpha=0.85)
    for xi, (a, b) in enumerate(zip(evm, nom)):
        ax.text(xi - 0.2, a, f"{a:.2f}", ha="center",
                va="bottom" if a >= 0 else "top", fontsize=8)
        ax.text(xi + 0.2, b, f"{b:.2f}", ha="center",
                va="bottom" if b >= 0 else "top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_ylabel("℃")
    ax.set_title("③ 사건 원인 분해")
    ax.legend(fontsize=8)

    # ④ 캘리브레이션
    ax = fig.add_subplot(gs[2, 0])
    bins = np.array([0, .01, .03, .06, .12, .25, .5, 1.01])
    xs, ys, ns = [], [], []
    p = d["p"].to_numpy()
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (p >= lo) & (p < hi)
        if k.sum() >= 20:
            xs.append(p[k].mean())
            ys.append(ev[k].mean())
            ns.append(int(k.sum()))
    ax.plot([0, max(xs + [0.1])], [0, max(xs + [0.1])], ls="--", color="#888", lw=0.9,
            label="완벽 보정")
    ax.plot(xs, ys, marker="o", ms=5, lw=1.2, color="#1a5a9e", label="관측")
    for xi, yi, ni in zip(xs, ys, ns):
        ax.annotate(f"n={ni}", (xi, yi), fontsize=6.5, xytext=(3, 4),
                    textcoords="offset points")
    ax.set_xlabel("예측 확률")
    ax.set_ylabel("실제 발생률")
    ax.set_title("④ 캘리브레이션")
    ax.legend(fontsize=8)

    # ⑤ 예측 vs 실제 (여유)
    ax = fig.add_subplot(gs[2, 1])
    mu, sd = d["mu"].to_numpy(), d["sd"].to_numpy()
    ax.scatter(mu[~ev], np.zeros((~ev).sum()) + 1, s=2, alpha=0.12, color="#4f8cd9",
               label="평시")
    ax.scatter(mu[ev], np.zeros(ev.sum()), s=3, alpha=0.5, color="#c0392b", label="사건")
    ax.axvline(0, color="#c0392b", ls="--", lw=1.0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["사건", "평시"])
    ax.set_xlabel("예측 공급온도 μ [℃]")
    ax.set_title("⑤ 예측 공급온도의 사건/평시 분리")
    ax.legend(fontsize=8, loc="upper right")

    # ⑥ 리프트 곡선
    ax = fig.add_subplot(gs[2, 2])
    order = np.argsort(-p)
    cum = np.cumsum(ev[order]) / max(ev.sum(), 1)
    frac = np.arange(1, len(p) + 1) / len(p)
    ax.plot(frac * 100, cum * 100, lw=1.3, color="#1a5a9e", label="PINN")
    ax.plot([0, 100], [0, 100], ls="--", color="#888", lw=0.9, label="무작위")
    for f0 in (1, 5):
        i = int(len(p) * f0 / 100)
        ax.axvline(f0, color="#c0392b", ls=":", lw=0.8)
        ax.annotate(f"{f0}%: {cum[i]*100:.0f}%", (f0, cum[i] * 100), fontsize=7.5,
                    xytext=(6, -8), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("상위 위험 구간 [%]")
    ax.set_ylabel("잡은 사건 [%]")
    ax.set_title("⑥ 리프트 곡선")
    ax.legend(fontsize=8, loc="lower right")

    # ⑦ 히터 물리 — 학습된 ε·유량과 바이패스 분율
    ax = fig.add_subplot(gs[3, :2])
    for u, c in zip(spec["units"], ("#2e6b45", "#8b4513")):
        ax.plot(d.index, d[f"eps_{u}"], lw=0.6, color=c, label=f"ε_{u} (예측)")
        if f"eps_obs_{u}" in d:
            ax.plot(d.index, d[f"eps_obs_{u}"], lw=0.5, color=c, alpha=0.45,
                    ls="--", label=f"ε_{u} (실측)")
    ax.plot(d.index, d["beta"], lw=0.7, color="#9b6bbf", label="β 바이패스 (학습)")
    ax.plot(d.index, d["beta_obs"], lw=0.6, color="#c0392b", alpha=0.55, label="β 실측")
    ax.set_ylim(-0.05, 1.35)
    ax.set_ylabel("무차원")
    ax.set_title("⑦ 히터 블록 — ε 은 **관측값**이고 역산하지 않는다 (문서 10)\n"
                 "    T_out = T_in + ε·(T_bath − T_in),  혼합가중 w ∝ m_설계·ε", fontsize=9.5)
    ax.legend(loc="upper right", ncol=3, fontsize=7, framealpha=0.9)

    # ⑧ 동시점 물리 정합 — (Q3)(Q4) 가 실측을 맞히는가
    ax = fig.add_subplot(gs[3, 2])
    pts = [(f"out_obs_{u}", f"out_now_{u}", f"(Q3) T_out,{u}") for u in spec["units"]]
    pts.append(("hdr_obs", "hdr_now", "(Q4) T_hdr"))
    for (a, b, lab), c in zip(pts, ("#2e6b45", "#8b4513", "#1a5a9e")):
        if a in d and b in d:
            ax.scatter(d[a], d[b], s=1.5, alpha=0.12, color=c, label=lab)
    lo = float(min(d[[c for c in d if c.startswith(("out_obs", "hdr_obs"))]].min()))
    hi = float(max(d[[c for c in d if c.startswith(("out_obs", "hdr_obs"))]].max()))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color="#666")
    ax.set_xlabel("실측 [℃]")
    ax.set_ylabel("물리층 계산 [℃]")
    ax.set_title(f"⑧ 동시점 정합 (RMSE ℃)\n"
                 f"(Q4) 헤더 {meta['Q4_헤더RMSE']:.1f}   (Q6) 공급 {meta['Q6_공급RMSE']:.1f}\n"
                 f"    (Q3) 는 항등 — ε 정의와 같다", fontsize=9)
    leg = ax.legend(fontsize=7.5, loc="upper left", markerscale=6)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    out = config.OUTPUT_DIR / f"pinn_{train}.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    log.info("저장: %s", out)
    return out


def main() -> None:
    for t in config.TRAINS:
        plot(t)


if __name__ == "__main__":
    main()
