"""검정 결과 시각화 — 한 장에 "식이 성립하나"의 증거를 모은다."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import style
from . import config, data, model as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ORDER = ["eps_weight", "eps_weight_b", "hotmax", "valve_only", "duty_weight", "equal",
         "equal_b", "eps_shuffle", "eps_shuffle_b", "free", "ntu_gated",
         "ntu_series", "ntu_series_b", "ntu_flat", "regime_gate"]
LABEL = {"hotmax": "max(T_out)\n(현행)", "valve_only": "밸브만\n(상류=상수)",
         "equal": "균등 가중", "design": "설계유량 가중", "eps_shuffle": "ε 시간치환\n(영가설)",
         "free": "자유 가중\n(신경망)", "regime_gate": "regime 게이트",
         "ntu_gated": "ε-NTU\n+차단게이트", "ntu_series": "ε-NTU\n(직렬 U)",
         "ntu_flat": "ε-NTU\n(상수 U)", "duty_weight": "버너 duty\n가중",
         "eps_weight": "ε 비례 가중\n(부호 반대)",
         "eps_weight_b": "ε 비례\n+매설", "equal_b": "균등\n+매설",
         "eps_shuffle_b": "ε 치환\n+매설(영가설)", "ntu_series_b": "ε-NTU\n+매설"}
PHYS = {"ntu_series", "ntu_flat", "ntu_gated", "ntu_series_b"}   # 식대로 역산 = 빨강
GOOD = {"eps_weight", "eps_weight_b"}                 # 부호를 뒤집은 경험식 = 초록


def plot(train: str, mix: pd.DataFrame, ind: pd.DataFrame, slo: pd.DataFrame):
    style.apply()
    units = config.TRAINS[train]["units"]
    d = data.build(train)
    tr, te = data.split(d)
    s = np.linspace(0, len(d) - 1, min(len(d), 120_000)).astype(int)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3,
                          left=0.06, right=0.985, top=0.855, bottom=0.055)
    fig.suptitle(f"히터 열전달식 검정 — 계열 {train} ({'+'.join(units)})   "
                 f"1분 원해상도 {len(d):,}행,  홀드아웃 {config.TEST_FROM}~",
                 fontsize=14, y=0.975)

    m = mix[mix["계열"] == train].set_index("가중")
    fig.text(0.06, 0.945,
             f"합류점 홀드아웃 R²:  ε-NTU 역산 {m.loc['ntu_series','홀드아웃_R2']:+.3f}   "
             f"영가설(ε 시간치환) {m.loc['eps_shuffle','홀드아웃_R2']:+.3f}   "
             f"균등 {m.loc['equal','홀드아웃_R2']:+.3f}   "
             f"ε 비례(부호 반대) {m.loc['eps_weight','홀드아웃_R2']:+.3f}\n"
             f"매설 구간(문서 11 §2-2)을 넣어도: ε-NTU {m.loc['ntu_series_b','홀드아웃_R2']:+.3f} "
             f"(그 영가설 {m.loc['eps_shuffle_b','홀드아웃_R2']:+.3f})   "
             f"ε 비례 {m.loc['eps_weight_b','홀드아웃_R2']:+.3f} — **기여 없음**\n"
             f"결론: 혼합식 (Q4) 는 성립한다(R² 최대 {m['홀드아웃_R2'].max():+.2f}). "
             f"그러나 유량 가중은 ε-NTU 가 말하는 것과 **부호가 반대**다 — "
             f"ε 이 높은 쪽이 유량이 많다.\n"
             f"        ε≈0 은 '대유량'이 아니라 '무유량'이다(정지 히터의 출구 열전대가 "
             f"정체 가스를 읽는다). 역산은 그걸 항상 대유량으로 읽는다.",
             fontsize=9.5, va="top", color="#333",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fdf0ee", ec="#d9b9b9"))

    # ① 가중방식별 홀드아웃 R²
    ax = fig.add_subplot(gs[0, :2])
    v = [float(m.loc[k, "홀드아웃_R2"]) for k in ORDER]
    cols = ["#c0392b" if k in PHYS else "#2e6b45" if k in GOOD
            else "#7f7f7f" if k.startswith("eps_shuffle") else "#1a5a9e" for k in ORDER]
    bars = ax.bar(range(len(ORDER)), v, color=cols, alpha=0.85)
    ax.axhline(float(m.loc["eps_shuffle", "홀드아웃_R2"]), color="#7f7f7f", ls="--", lw=1.1,
               label="영가설 (ε 시간치환)")
    ax.axhline(0, color="#666", lw=0.8)
    for b, x in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, x, f"{x:+.2f}", ha="center",
                va="bottom" if x >= 0 else "top", fontsize=8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=6.5)
    ax.set_ylabel("홀드아웃 R²")
    ax.set_title("① 합류점 예측 — 가중방식만 바꾼 대조군 (빨강 = 물리식 기반)")
    ax.legend(fontsize=8, loc="lower left")

    # ② β(밸브) 학습맵 vs 실측 β
    ax = fig.add_subplot(gs[0, 2])
    hot = d[[f"out_{u}" for u in units]].max(axis=1)
    spread = hot - d["t_in"]
    b_obs = ((hot - d["hdr"]) / spread.where(spread > 5)).clip(-0.2, 1.2)
    ax.scatter(d["valve"].to_numpy()[s], b_obs.to_numpy()[s], s=1.5, alpha=0.06,
               color="#1a5a9e", label="실측 β")
    q = np.quantile(d["valve"], np.linspace(0.01, 0.99, 25))
    bm = [float(b_obs[(d["valve"] >= lo) & (d["valve"] < hi)].median())
          for lo, hi in zip(q[:-1], q[1:])]
    ax.plot((q[:-1] + q[1:]) / 2, bm, lw=1.6, color="#c0392b", label="실측 중앙")
    ax.set_xlabel("바이패스 밸브 개도")
    ax.set_ylabel("β")
    ax.set_title("② 바이패스 분율은 밸브를 따른다")
    ax.legend(fontsize=8, markerscale=6)

    # ③ ε≈0 의 2가 모호성
    ax = fig.add_subplot(gs[1, 0])
    u0 = units[0]
    for reg, c in (("flow", "#1a5a9e"), ("isolated", "#c0392b"), ("lowflow", "#2e6b45")):
        k = (d[f"regime_{u0}"] == reg).to_numpy()[s]
        if k.sum():
            ax.hist(d[f"eps_{u0}"].to_numpy()[s][k], bins=60, range=(0, 1), alpha=0.55,
                    color=c, label=f"{reg} ({100*k.mean():.0f}%)")
    ax.set_xlabel(f"ε_{u0}")
    ax.set_ylabel("행 수")
    ax.set_title(f"③ ε≈0 은 '대유량'과 '차단'을 구분하지 못한다\n"
                 f"    역산은 항상 대유량으로 읽는다", fontsize=9.5)
    ax.legend(fontsize=8)

    # ④ 헤더온도는 어느 출구를 따라가나
    ax = fig.add_subplot(gs[1, 1])
    reg = d[[f"regime_{u}" for u in units]]
    isflow = (reg == "flow").to_numpy()
    one = isflow.sum(1) == 1
    o = d[[f"out_{u}" for u in units]].to_numpy()
    act = np.where(isflow[:, 0], o[:, 0], o[:, 1])
    hdr = d["hdr"].to_numpy()
    for lab, x, c in (("max(T_out)", o.max(1), "#c0392b"),
                      ("min(T_out)", o.min(1), "#1a5a9e")):
        k = one & np.isin(np.arange(len(d)), s)
        ax.hist((hdr - x)[k], bins=70, range=(-35, 25), alpha=0.55, color=c,
                label=f"{lab} (중앙 {np.median((hdr - x)[one]):+.1f}℃)")
    ax.axvline(0, color="#666", lw=1.0)
    inside = ((hdr >= np.minimum(o.min(1), d["t_in"].to_numpy())) & (hdr <= o.max(1)))
    ax.set_xlabel("T_hdr − 출구온도 [℃]")
    ax.set_title(f"④ 헤더는 두 출구 **사이**에 있다 (혼합식 성립 가능 {100*inside.mean():.0f}%)\n"
                 f"    → 두 히터가 동시에 기여한다. regime 라벨 1대는 {100*one.mean():.0f}%",
                 fontsize=9.5)
    ax.legend(fontsize=8)

    # ⑤ 무모수 독립성 검정
    ax = fig.add_subplot(gs[1, 2])
    ii = ind[ind["계열"] == train]
    x = np.arange(len(ii))
    ax.bar(x - 0.2, ii["|ρ|중앙"], 0.38, color="#1a5a9e", label="실측 |ρ| 중앙")
    ax.bar(x + 0.2, ii["대조군|ρ|중앙"], 0.38, color="#7f7f7f", label="치환 대조군")
    ax.plot(x, ii["부호일관성"], "o--", ms=5, color="#c0392b", label="부호 일관성")
    ax.axhline(0.5, color="#c0392b", ls=":", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.히터}\n{r.대상}" for r in ii.itertuples()], fontsize=7.5)
    ax.set_title("⑤ ε 은 T_bath·T_in 과 무관한가\n    (식이 맞으면 |ρ|→0, 부호일관성→0.5)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    # ⑥ 기울기 검정
    ax = fig.add_subplot(gs[2, 0])
    ss = slo[slo["계열"] == train]
    x = np.arange(len(ss))
    ax.bar(x - 0.2, ss["기울기중앙"], 0.38, color="#1a5a9e", label="∂T_out/∂T_bath")
    ax.bar(x + 0.2, ss["ε중앙"], 0.38, color="#2e6b45", label="ε (식의 예측)")
    for i, r in enumerate(ss.itertuples()):
        ax.text(i, max(r.기울기중앙, r.ε중앙) + 0.02, f"Δ{r.차이중앙:+.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ss["히터"])
    ax.set_ylim(0, 1.0)
    ax.set_title("⑥ (Q3) 기울기 예측 — 자유 모수 없음", fontsize=9.5)
    ax.legend(fontsize=8)

    # ⑦ 잔차 구조
    ax = fig.add_subplot(gs[2, 1:])
    rs = pd.read_csv(config.OUTPUT_DIR / "mixing_residual.csv")
    rs = rs[rs["계열"] == train].set_index("가중")
    cc = [c for c in rs.columns if c.startswith("잔차ρ_") and rs[c].notna().all()]
    x = np.arange(len(cc))
    for i, (w, c) in enumerate((("ntu_series", "#c0392b"), ("equal", "#7f7f7f"),
                                ("hotmax", "#1a5a9e"))):
        if w in rs.index:
            ax.bar(x + (i - 1) * 0.27, rs.loc[w, cc].to_numpy(dtype=float), 0.26,
                   color=c, alpha=0.85, label=LABEL[w].replace("\n", " "))
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("잔차ρ_", "") for c in cc], fontsize=8, rotation=20)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("⑦ 홀드아웃 잔차 구조 — 식이 맞으면 전부 0 근처여야 한다")
    ax.legend(fontsize=8)

    out = config.OUTPUT_DIR / f"physics_{train}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log.info("저장: %s", out)


def main() -> None:
    mix = pd.read_csv(config.OUTPUT_DIR / "mixing_test.csv")
    ind = pd.read_csv(config.OUTPUT_DIR / "verify_independence.csv")
    slo = pd.read_csv(config.OUTPUT_DIR / "verify_slope.csv")
    for t in config.TRAINS:
        plot(t, mix, ind, slo)


if __name__ == "__main__":
    main()
