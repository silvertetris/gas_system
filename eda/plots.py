"""태그 하나당 PNG 하나. 설비·부위·역할을 그림에 실어 그림만 봐도 무슨 태그인지 알게 한다.

한 장의 구성 (2×3):
    ┌──────────────────────────────────────────────┐
    │ 제목: 태그 · 설비 / 절 · 설명                  │
    │ 메타 박스: 측정량·계기기능·역할·신호·배선·상태   │
    ├───────────────────┬──────────────┬───────────┤
    │ 전체 시계열(일 단위) │ 히스토그램     │ 통계표     │
    │ + 정기점검 7건 표시  │              │           │
    ├───────────────────┼──────────────┼───────────┤
    │ 월별 커버리지        │ 연도별 박스플롯 │ 일중 패턴  │
    └───────────────────┴──────────────┴───────────┘
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, style, tagmeta

logger = logging.getLogger(__name__)


def _stats(s: pd.Series) -> list[tuple[str, str]]:
    v = s.dropna()
    nz = v[v != 0]
    rows = [
        ("관측 행수", f"{len(v):,}"),
        ("결측률", f"{100 * (1 - len(v) / max(len(s), 1)):.2f}%"),
        ("0 아닌 값", f"{len(nz):,}"),
        ("기간", f"{v.index.min():%Y-%m-%d} ~ {v.index.max():%Y-%m-%d}" if len(v) else "-"),
        ("평균", f"{v.mean():.3f}" if len(v) else "-"),
        ("중앙값", f"{v.median():.3f}" if len(v) else "-"),
        ("표준편차", f"{v.std():.3f}" if len(v) else "-"),
        ("최소 / 최대", f"{v.min():.2f} / {v.max():.2f}" if len(v) else "-"),
        ("p1 / p99", f"{v.quantile(.01):.2f} / {v.quantile(.99):.2f}" if len(v) else "-"),
        ("고유값 수", f"{v.nunique():,}"),
    ]
    if len(nz) > 1:
        d = np.abs(np.diff(np.sort(nz.unique())))
        d = d[d > 0]
        if len(d):
            rows.append(("분해능(최소차)", f"{d.min():.4f}"))
    if len(v) > 1:
        ch = (v.diff() != 0).mean()
        rows.append(("값 변화 비율", f"{ch:.3f}"))
    return rows


def plot_tag(tag: str, s: pd.Series, out_dir) -> "object":
    style.apply()
    meta = tagmeta.describe(tag)
    v = s.dropna()
    nz = v[v != 0]
    dead = len(nz) < config.DEAD_NONZERO_MAX

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1.1, 1.0], height_ratios=[1, 1],
                          hspace=0.38, wspace=0.26,
                          left=0.055, right=0.985, top=0.80, bottom=0.075)

    # --- 제목·메타
    head = f"{tag}"
    sub = f"{meta['설비']}"
    if meta["절"]:
        sub += f"  ›  {meta['절']}"
    if meta["호기"]:
        sub += f"   [{meta['호기']}]"
    fig.text(0.055, 0.955, head, fontsize=19, fontweight="bold", va="top")
    fig.text(0.055, 0.915, sub, fontsize=10.5, color="#333", va="top")
    fig.text(0.055, 0.888, meta["설명"], fontsize=11, color="#0b4f8a", va="top")

    info = (f"측정량 {meta['측정량'] or '-'}   |   계기기능 {meta['계기기능'] or '-'}   |   "
            f"역할 {meta['역할'] or '-'}   |   신호 {meta['신호'] or '-'}   |   "
            f"배선 {meta['배선'] or '-'}   |   사전상태 {meta['상태'] or '-'}")
    fig.text(0.055, 0.858, info, fontsize=8.6, color="#444", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f2f5f8", ec="#c8d2dc"))
    if meta["주의"]:
        fig.text(0.055, 0.826, meta["주의"], fontsize=8.6, color="#8a2b0b", va="top",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fdf0ea", ec="#e0b3a0"))
    if dead:
        fig.text(0.985, 0.955, "DEAD\n(0 아닌 값 희소)", fontsize=11, color="#a11", va="top",
                 ha="right", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fbeaea", ec="#d99"))

    # --- ① 전체 시계열
    ax = fig.add_subplot(gs[0, 0])
    if len(v):
        d = v.resample(config.RESAMPLE)
        med, lo, hi = d.median(), d.quantile(.05), d.quantile(.95)
        ax.fill_between(med.index, lo, hi, alpha=0.22, color="#4f8cd9", lw=0, label="p5~p95")
        ax.plot(med.index, med, lw=0.7, color="#1a5a9e", label=f"중앙({config.RESAMPLE})")
        for m in config.MAINTENANCE:
            ax.axvline(pd.Timestamp(m), color="#c0392b", ls=":", lw=0.9, alpha=0.75)
        ax.legend(loc="upper left", framealpha=0.85)
    ax.set_title(f"전체 시계열 — 점선 = 정기점검 {len(config.MAINTENANCE)}건")
    ax.set_ylabel("값")

    # --- ② 분포
    ax = fig.add_subplot(gs[0, 1])
    if len(v):
        x = v.sample(min(config.SAMPLE_MAX, len(v)), random_state=0) if len(v) > config.SAMPLE_MAX else v
        ax.hist(x, bins=config.HIST_BINS, color="#4f8cd9", edgecolor="white", lw=0.3)
        ax.axvline(v.median(), color="#c0392b", lw=1.2, label=f"중앙 {v.median():.2f}")
        ax.legend(framealpha=0.85)
        ax.set_yscale("log")
    ax.set_title("값 분포 (y 로그)")

    # --- ③ 통계표
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    rows = _stats(s)
    tbl = ax.table(cellText=[[k, val] for k, val in rows], colLabels=["항목", "값"],
                   cellLoc="left", loc="upper left", colWidths=[0.52, 0.48])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.24)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d5dce3")
        if r == 0:
            cell.set_facecolor("#e8eef4")
            cell.set_text_props(fontweight="bold")

    # --- ④ 월별 커버리지
    ax = fig.add_subplot(gs[1, 0])
    cov = s.notna().resample("MS").mean() * 100
    ax.fill_between(cov.index, 0, cov.to_numpy(), color="#5b9e6f", alpha=0.55, lw=0)
    ax.plot(cov.index, cov, lw=0.6, color="#2e6b45")
    ax.set_ylim(0, 105)
    ax.set_title("월별 실측 커버리지 [%]")
    ax.set_ylabel("%")

    # --- ⑤ 연도별 분포
    ax = fig.add_subplot(gs[1, 1])
    if len(v):
        yrs = sorted(v.index.year.unique())
        dat = [v[v.index.year == y].to_numpy() for y in yrs]
        dat = [d if len(d) else np.array([np.nan]) for d in dat]
        # matplotlib 3.9+ 에서 `labels` → `tick_labels` 로 이름이 바뀌었다
        bp = ax.boxplot(dat, tick_labels=[str(y)[2:] for y in yrs], showfliers=False,
                        patch_artist=True, medianprops=dict(color="#c0392b"))
        for b in bp["boxes"]:
            b.set(facecolor="#cfe0f2", edgecolor="#5580a8", lw=0.6)
    ax.set_title("연도별 분포")
    ax.tick_params(axis="x", rotation=90)

    # --- ⑥ 일중 패턴
    ax = fig.add_subplot(gs[1, 2])
    if len(v):
        h = v.groupby(v.index.hour)
        m2, l2, h2 = h.median(), h.quantile(.25), h.quantile(.75)
        ax.fill_between(m2.index, l2, h2, alpha=0.25, color="#9b6bbf", lw=0)
        ax.plot(m2.index, m2, marker="o", ms=2.5, lw=0.9, color="#6b3f8f")
        ax.set_xticks(range(0, 24, 4))
    ax.set_title("일중 패턴 (시각별 중앙·IQR)")
    ax.set_xlabel("시각")

    out = out_dir / f"{tag.replace('/', '_')}.png"
    fig.savefig(out, dpi=config.FIG_DPI)
    plt.close(fig)
    return out
