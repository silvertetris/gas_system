"""비교 요약 — 표 + 그림. `variants_run` 결과(…/v_<조합>/metrics_seeds_*.csv)를 모두 모은다.

두 가지 비교:
  1. **정제 조합** — 같은 구조·타깃 안에서 없음 / KF / AE / EKF / 셋 다 (구조·타깃마다 그림 한 장)
  2. **구조** — 같은 타깃·정제 조합 안에서 soft(물리 = 손실 제약) / hard(출력이 물리식을 거침) /
     nn(물리 없는 같은 신경망). 같은 시드끼리 짝비교.

형태: 범주는 **x축**(정제 조합), 구조 3개는 팔레트 첫 세 슬롯(blue·orange·aqua — 전쌍 검증 통과)으로,
GBM 은 잉크색 ◆ + 범례 라벨. 점 = 시드별 값, 굵은 표식 = 평균, 막대 = ±1 표준편차. 지표마다 따로
축(이중축 없음). aqua 는 밝은 배경 대비가 3:1 미만이라 범례·축 라벨로 식별을 보장한다(표 뷰: csv).
표 원자료: variants_{raw,summary,pairs}.csv · arch_pairs.csv.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import style
from . import config, tune

log = logging.getLogger(__name__)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
ARCH_COLOR = {"soft": BLUE, "hard": ORANGE, "nn": AQUA, "lstm": YELLOW, "softw": MAGENTA}   # 팔레트 고정 순서 1~5
ARCH_NAME = {"soft": "soft PINN (물리=손실)", "hard": "hard PINN (출력=물리식)", "nn": "nn (물리 없음)",
             "lstm": "LSTM (시계열, 물리 없음)", "softw": "soft PINN (물리 가중 0.3~30)"}
TARGET_NAME = {"min_raw": "분 단위 최저 (이상값 포함)", "min": "분 단위 최저 (센서 이상값 제거)",
               "mean": "1시간 평균 최저"}
PANELS = [("CV_AUC", None, "CV AUC (개발구간 5-fold, 최적 trial)"),
          ("PINN_전체_AUC", "GBM_전체_AUC", "시험 AUC"),
          ("PINN_전체_리프트1", "GBM_전체_리프트1", "시험 리프트 1% (배)"),
          ("PINN_전체_Brier개선%", "GBM_전체_Brier개선%", "시험 Brier 개선 (%)")]
KEYS = ["구조", "타깃", "계열", "정제"]


def collect() -> pd.DataFrame:
    frames = []
    for p in sorted(config.OPTUNA_DIR.rglob("metrics_seeds_*.csv")):
        top = p.relative_to(config.OPTUNA_DIR).parts[0]
        if top == "smoke" or top.startswith("_") or not p.parent.name.startswith("v_"):
            continue
        d = pd.read_csv(p)
        d["구조"] = d["구조"] if "구조" in d else "hard"
        d["타깃"] = d["타깃"] if "타깃" in d else "min_raw"
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"결과 없음: {config.OPTUNA_DIR}/**/v_*/metrics_seeds_*.csv")
    return pd.concat(frames, ignore_index=True)


def _paired(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> pd.Series:
    return (a.set_index("시드")[metric] - b.set_index("시드")[metric]).dropna()


def tables(raw: pd.DataFrame):
    keys = ["PINN_전체_AUC", "PINN_전체_리프트1", "PINN_전체_리프트5", "PINN_전체_Brier개선%",
            "PINN_유량관측_AUC", "PINN_유량미관측_AUC", "공급온도예측RMSE", "μ_JT", "UA_배관_kW/K"]
    g = raw.groupby(KEYS, sort=False)
    summ = g[keys].agg(["mean", "std"]).round(3)
    summ.columns = [f"{a}_{'평균' if b == 'mean' else '표준편차'}" for a, b in summ.columns]
    first = g.agg(CV_AUC=("CV_AUC", "first"), GBM_전체_AUC=("GBM_전체_AUC", "first"),
                  GBM_전체_리프트1=("GBM_전체_리프트1", "first"),
                  GBM_전체_Brier개선=("GBM_전체_Brier개선%", "first"), 시드수=("시드", "count"),
                  시험사건=("시험사건", "first"))
    hp = g[[c for c in raw.columns if c.startswith("hp_")]].first()
    summ = first.join(summ).join(hp).reset_index()

    pairs = []
    for (arch, target, t), d in raw.groupby(["구조", "타깃", "계열"]):
        base = d[d["정제"] == "없음"]
        for v in tune.VARIANTS[1:]:
            cur = d[d["정제"] == v]
            if cur.empty or base.empty:
                continue
            row = {"구조": arch, "타깃": target, "계열": t, "조합": v}
            for m in ("PINN_전체_AUC", "PINN_전체_리프트1", "PINN_전체_Brier개선%"):
                diff = _paired(cur, base, m)
                row[f"Δ{m}_평균"] = round(float(diff.mean()), 4)
                row[f"Δ{m}_개선시드"] = f"{int((diff > 0).sum())}/{len(diff)}"
            row["ΔCV_AUC"] = round(float(cur["CV_AUC"].iloc[0] - base["CV_AUC"].iloc[0]), 4)
            row["ΔGBM_AUC"] = round(float(cur["GBM_전체_AUC"].iloc[0] - base["GBM_전체_AUC"].iloc[0]), 4)
            pairs.append(row)

    arch_pairs = []
    for (target, t, v), d in raw.groupby(["타깃", "계열", "정제"]):
        for a, b in (("soft", "nn"), ("hard", "nn"), ("soft", "hard"), ("lstm", "nn"), ("soft", "lstm"),
                     ("softw", "nn"), ("softw", "soft"), ("softw", "hard"), ("softw", "lstm")):
            da, db = d[d["구조"] == a], d[d["구조"] == b]
            if da.empty or db.empty:
                continue
            row = {"타깃": target, "계열": t, "정제": v, "비교": f"{a} − {b}"}
            for m in ("PINN_전체_AUC", "PINN_전체_리프트1", "PINN_전체_Brier개선%"):
                diff = _paired(da, db, m)
                row[f"Δ{m}_평균"] = round(float(diff.mean()), 4)
                row[f"Δ{m}_개선시드"] = f"{int((diff > 0).sum())}/{len(diff)}"
            row["ΔCV_AUC"] = round(float(da["CV_AUC"].iloc[0] - db["CV_AUC"].iloc[0]), 4)
            arch_pairs.append(row)
    return summ, pd.DataFrame(pairs), pd.DataFrame(arch_pairs)


def _axes(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _dots(ax, x, vals, color, label=None):
    vals = np.asarray(vals, float)
    if not len(vals):
        return
    jit = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.035
    ax.scatter(x + jit, vals, s=14, color=color, alpha=0.35, lw=0, zorder=2)
    ax.errorbar(x, np.nanmean(vals), yerr=np.nanstd(vals, ddof=1) if len(vals) > 1 else 0, fmt="o", ms=7,
                color=color, elinewidth=2, capsize=0, zorder=3, label=label, markeredgecolor=SURF, markeredgewidth=1)


def plot_refine(raw: pd.DataFrame, out, arch: str, target: str):
    """정제 조합 비교 (구조·타깃 하나)."""
    trains = [t for t in config.TRAINS if t in set(raw["계열"])]
    order = [v for v in tune.VARIANTS if v in set(raw["정제"])]
    n_seed = int(raw.groupby(["계열", "정제"])["시드"].count().max())
    ncol = len(PANELS) + 1
    H = 4.7 * len(trains) + 1.6
    fig = plt.figure(figsize=(4.6 * ncol, H), facecolor=SURF)
    # 제목·부제·패널 범례 자리를 **인치로** 확보한다 (행 수가 적으면 비율 여백이 모자라 겹쳤다)
    gs = fig.add_gridspec(len(trains), ncol, hspace=0.6, wspace=0.3, left=0.04, right=0.99,
                          top=1 - 1.55 / H, bottom=0.7 / H)
    fig.suptitle(f"정제 조합별 — {ARCH_NAME[arch]} · 타깃 {TARGET_NAME[target]} · 조합마다 Optuna(8:2, 5-fold) · "
                 f"시드 {n_seed}개", fontsize=14, color=INK, y=1 - 0.3 / H)
    fig.text(0.04, 1 - 0.7 / H, "점 = 시드별 값 · 굵은 표식 = 평균 · 막대 = ±1 표준편차 · ◆ = 같은 입력 GBM   |   "
             "마지막 열: 같은 시드끼리 뺀 (조합 − 없음) 시험 AUC, 0 위면 개선", fontsize=9.5, color=INK2)
    x = np.arange(len(order))
    for r, t in enumerate(trains):
        d = raw[raw["계열"] == t]
        for c, (pm, gm, label) in enumerate(PANELS):
            ax = fig.add_subplot(gs[r, c]); _axes(ax)
            for i, v in enumerate(order):
                vals = d.loc[d["정제"] == v, pm].to_numpy(float)
                _dots(ax, i, vals[:1] if pm == "CV_AUC" else vals, BLUE, "PINN" if i == 0 else None)
                if gm and (d["정제"] == v).any():
                    ax.scatter(i + 0.28, d.loc[d["정제"] == v, gm].iloc[0], marker="D", s=30, color=INK2,
                               zorder=3, label="GBM" if i == 0 else None)
            ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8, color=INK2, rotation=15)
            ax.set_title(f"{t} — {label}", fontsize=10, color=INK, pad=18 if c == 1 else 6)
            if c == 1:
                ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left",
                          bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
        ax = fig.add_subplot(gs[r, len(PANELS)]); _axes(ax)
        base = d[d["정제"] == "없음"]
        for i, v in enumerate(order):
            if v == "없음" or base.empty:
                continue
            diff = _paired(d[d["정제"] == v], base, "PINN_전체_AUC").to_numpy()
            if not len(diff):
                continue
            _dots(ax, i, diff, BLUE)
            ax.annotate(f"{int((diff > 0).sum())}/{len(diff)} 개선", (i, diff.mean()), xytext=(8, 0),
                        textcoords="offset points", fontsize=7.5, color=INK2, va="center")
        ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8, color=INK2, rotation=15)
        ax.set_xlim(0.4, len(order) - 0.4)
        ax.set_title(f"{t} — 시험 AUC 차이 (조합 − 없음)", fontsize=10, color=INK)
    fig.savefig(out, dpi=110, facecolor=SURF)
    plt.close(fig)
    return out


def plot_arch(raw: pd.DataFrame, out, metric: str = "PINN_전체_AUC", gbm: str | None = "GBM_전체_AUC",
              label: str = "시험 AUC"):
    """구조 비교 — 행 = 계열, 열 = 타깃, x = 정제 조합, 색 = 구조. 마지막 열 = soft − nn 짝차이."""
    r0 = raw[raw["타깃"] != "min_raw"]
    targets = [t for t in ("min", "mean") if t in set(r0["타깃"])]
    trains = [t for t in config.TRAINS if t in set(r0["계열"])]
    archs = [a for a in ("softw", "soft", "hard", "nn", "lstm") if a in set(r0["구조"])]
    if not targets or len(archs) < 2:
        return None
    order = [v for v in tune.VARIANTS if v in set(r0["정제"])]
    ncol = len(targets) * 2
    H = 4.9 * len(trains) + 1.9
    fig = plt.figure(figsize=(6.2 * ncol, H), facecolor=SURF)
    gs = fig.add_gridspec(len(trains), ncol, hspace=0.62, wspace=0.25, left=0.035, right=0.995,
                          top=1 - 1.85 / H, bottom=0.7 / H)
    fig.suptitle(f"구조 비교 — 물리가 정확도를 올리나 · {label}", fontsize=14, color=INK, y=1 - 0.3 / H)
    fig.text(0.035, 1 - 0.7 / H, "색 = 구조 · 점 = 시드별 · 굵은 표식 = 평균 ± 1 표준편차 · ◆ = 같은 입력 GBM   |   "
             "짝차이 열: 같은 시드·같은 정제에서 (soft − nn), (hard − nn). 0 위면 물리가 도움", fontsize=9.5, color=INK2)
    x = np.arange(len(order))
    off = {a: (k - (len(archs) - 1) / 2) * (0.8 / max(len(archs), 1)) for k, a in enumerate(archs)}
    for r, t in enumerate(trains):
        for c, tg in enumerate(targets):
            d = r0[(r0["계열"] == t) & (r0["타깃"] == tg)]
            ax = fig.add_subplot(gs[r, 2 * c]); _axes(ax)
            for i, v in enumerate(order):
                for a in archs:
                    _dots(ax, i + off[a], d.loc[(d["정제"] == v) & (d["구조"] == a), metric],
                          ARCH_COLOR[a], ARCH_NAME[a] if i == 0 else None)
                if gbm and ((d["정제"] == v)).any():
                    ax.scatter(i + 0.42, d.loc[d["정제"] == v, gbm].iloc[0], marker="D", s=30, color=INK2,
                               zorder=3, label="GBM" if i == 0 else None)
            ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8, color=INK2, rotation=15)
            ax.set_xlim(-0.5, len(order) - 0.3)
            ax.set_title(f"{t} · 타깃 {TARGET_NAME[tg]}", fontsize=10, color=INK, pad=30)
            ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                      ncol=4, borderaxespad=0.2)

            ax = fig.add_subplot(gs[r, 2 * c + 1]); _axes(ax)
            for i, v in enumerate(order):
                dv = d[d["정제"] == v]
                others = [a for a in ("softw", "soft", "hard", "lstm") if a in archs]
                for k, a in enumerate(others):
                    if "nn" not in archs:
                        continue
                    diff = _paired(dv[dv["구조"] == a], dv[dv["구조"] == "nn"], metric).to_numpy()
                    if len(diff):
                        xx = i + (k - (len(others) - 1) / 2) * 0.25
                        _dots(ax, xx, diff, ARCH_COLOR[a], f"{a} − nn" if i == 0 else None)
                        ax.annotate(f"{int((diff > 0).sum())}/{len(diff)}", (xx, diff.mean()), xytext=(0, 9),
                                    textcoords="offset points", fontsize=7, color=INK2, ha="center")
            ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
            ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8, color=INK2, rotation=15)
            ax.set_xlim(-0.5, len(order) - 0.5)
            if not ax.get_legend_handles_labels()[0]:
                ax.text(0.5, 0.5, "짝지을 nn 결과 없음", transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_title(f"{t} · {TARGET_NAME[tg]} — 짝차이 (− nn)", fontsize=10, color=INK, pad=18)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                          ncol=2, borderaxespad=0.2)
    fig.savefig(out, dpi=105, facecolor=SURF)
    plt.close(fig)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    style.apply()
    raw = collect()
    summ, pairs, arch_pairs = tables(raw)
    o = config.OPTUNA_DIR
    raw.to_csv(o / "variants_raw.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(o / "variants_summary.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(o / "variants_pairs.csv", index=False, encoding="utf-8-sig")
    arch_pairs.to_csv(o / "arch_pairs.csv", index=False, encoding="utf-8-sig")
    for (arch, target), d in raw.groupby(["구조", "타깃"]):
        out = config.variant_dir("auto", False, arch, target) / "variants_compare.png"
        log.info("저장: %s", plot_refine(d, out, arch, target))
    for metric, gbm, label, name in (("PINN_전체_AUC", "GBM_전체_AUC", "시험 AUC", "arch_compare_auc.png"),
                                     ("PINN_전체_리프트1", "GBM_전체_리프트1", "시험 리프트 1% (배)", "arch_compare_lift1.png"),
                                     ("CV_AUC", None, "CV AUC", "arch_compare_cv.png")):
        p = plot_arch(raw, o / name, metric, gbm, label)
        if p:
            log.info("저장: %s", p)
    pd.set_option("display.width", 260)
    show = KEYS + ["시드수", "CV_AUC", "PINN_전체_AUC_평균", "PINN_전체_AUC_표준편차", "GBM_전체_AUC",
                   "PINN_전체_리프트1_평균", "GBM_전체_리프트1", "PINN_전체_Brier개선%_평균"]
    print(summ[[c for c in show if c in summ.columns]].to_string(index=False))
    if len(arch_pairs):
        print(arch_pairs.to_string(index=False))


if __name__ == "__main__":
    main()
