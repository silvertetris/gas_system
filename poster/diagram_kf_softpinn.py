"""KF-soft PINN 구조도 (포스터용 PNG, 300 dpi).

흐름: ① 운전 데이터 → ② 칼만필터 정제 → ③ 신경망(공유 은닉층 → T61 헤드 · 상태 헤드) → ④ 물리층(열전달식)
      → ⑤ 손실(데이터 NLL + 물리 잔차 + 정합 잔차, Optuna 가중치) → ⑥ 빙결 확률 · 경보 / 물리 분해.
실선 = 예측 경로, 점선 = 학습할 때만 쓰는 손실 경로. 색: 데이터 회색 · KF 청록 · 신경망 파랑 · 물리 주황 · 손실 보라.

    .venv/bin/python poster/diagram_kf_softpinn.py      # → ttttt/KF_softPINN_구조도.png
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ttttt" / "KF_softPINN_구조도.png"

INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#ffffff"
C = {  # (테두리, 채움)
    "data": ("#898781", "#f0efec"), "kf": ("#1baf7a", "#e3f4ee"), "nn": ("#2a78d6", "#e3effc"),
    "phys": ("#eb6834", "#fde9df"), "loss": ("#4a3aa7", "#ece9f8"), "out": ("#c98500", "#fff4d6"),
}
KR = "Noto Sans CJK KR"


def box(ax, x, y, w, h, kind, r=1.2, lw=2.2):
    edge, fill = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=edge, lw=lw))


def head(ax, x, y, w, text, kind, size=17):
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=size, fontweight="bold", color=C[kind][0], family=KR)


def t(ax, x, y, s, size=13, color=INK, ha="left", bold=False, family=KR):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, family=family,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color=INK2, dashed=False, rad=0.0, lw=2.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, color=color, lw=lw,
                                 linestyle=(0, (5, 4)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def main():
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig = plt.figure(figsize=(18, 10.4), dpi=300, facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 104); ax.axis("off")

    t(ax, 90, 99.5, "KF-soft PINN 구조 — 칼만필터 정제 + 물리식 손실 제약 신경망", 24, ha="center", bold=True)
    t(ax, 90, 95.0, "6시간 이내 정압기 하류 가스 빙결 확률 예측 · 실선 = 예측 경로 · 점선 = 학습 시 손실 경로",
      14, INK2, ha="center")

    # ① 운전 데이터
    box(ax, 2, 12, 30, 78, "data")
    head(ax, 2, 86, 30, "① 운전 데이터 (1시간 단위)", "data")
    items = [("공급온도", r"$T_{61}$"), ("헤더온도", r"$T_{hdr}$"), ("입구온도", r"$T_{in}$"),
             ("압력강하", r"$\Delta P = P_{in}-P_{61}$"), ("가스 유량", r"$\dot m$ : 수조 열수지로 역산"), ("계절·시각", r"$\sin,\ \cos$")]
    for i, (k, v) in enumerate(items):
        yy = 76 - i * 10.6
        ax.add_patch(FancyBboxPatch((4, yy - 4), 26, 8, boxstyle="round,pad=0,rounding_size=0.8", fc=SURF,
                                    ec="#c3c2b7", lw=1.2))
        t(ax, 5.5, yy + 1.3, k, 13, INK2)
        t(ax, 5.5, yy - 1.8, v, 14, INK)

    # ② 칼만필터 정제
    box(ax, 40, 56, 30, 30, "kf")
    head(ax, 40, 82, 30, "② 칼만필터 정제", "kf")
    t(ax, 55, 75.5, r"$\hat{x}_t=\hat{x}_{t-1}+K\,(z_t-\hat{x}_{t-1})$", 15, ha="center")
    t(ax, 55, 69.5, r"$\nu_t=z_t-\hat{x}_{t-1}$", 15, ha="center")
    t(ax, 55, 63.5, "평활값  +  혁신(급변 신호)", 13, INK2, ha="center")
    t(ax, 55, 59.3, r"대상: $T_{61},\ T_{hdr},\ T_{in}$", 13, INK2, ha="center")

    # ③ 신경망
    box(ax, 78, 42, 30, 20, "nn")
    head(ax, 78, 57.5, 30, "③ 신경망 (공유 은닉층)", "nn")
    t(ax, 93, 50.5, "입력: 현재 계측 · 이력 · 유량", 12.5, INK2, ha="center")
    t(ax, 93, 46.3, "계절·시각 · KF 신호", 12.5, INK2, ha="center")
    box(ax, 78, 68, 30, 18, "nn")
    head(ax, 78, 82, 30, "T61 헤드 (직접 예측)", "nn", 15.5)
    t(ax, 93, 75.5, r"$\hat\mu,\ \hat\sigma$ : 6시간 내 최저 $T_{61}$", 14, ha="center")
    t(ax, 93, 71, r"$d\geq0$ : 순간 하강분", 13, INK2, ha="center")
    box(ax, 78, 12, 30, 24, "nn")
    head(ax, 78, 32, 30, "상태 헤드 (최저온 시각)", "nn", 15.5)
    t(ax, 93, 25, r"$\hat{\mathbf{s}}=(\hat T_{hdr},\ \Delta\hat P,\ \hat T_{in},\ \log\hat{\dot m})$", 14, ha="center")
    t(ax, 93, 18.5, "6시간 내 T61 이 가장 낮은 시각의 상태", 12, INK2, ha="center")
    t(ax, 93, 14.8, "평균 · 표준편차", 12, INK2, ha="center")

    # ④ 물리층
    box(ax, 116, 12, 34, 30, "phys")
    head(ax, 116, 38, 34, "④ 물리층 (열전달식)", "phys")
    t(ax, 133, 31, r"$f(\mathbf{s})=T_g+(T_{hdr}-\mu_{JT}\,\Delta P-T_g)\,e^{-N}$", 14, ha="center")
    t(ax, 133, 25, r"$N=\dfrac{UA}{\dot m\,c_p}+N_0,\qquad T_g=a+b\,T_{in}$", 14, ha="center")
    t(ax, 133, 18.5, "줄-톰슨 냉각 + 배관 열교환 (같은 시각 관계)", 12.5, INK2, ha="center")
    t(ax, 133, 14.8, r"학습 계수: $\mu_{JT},\ UA,\ N_0,\ a,\ b$", 12.5, INK2, ha="center")

    # ⑤ 손실
    box(ax, 116, 47, 62, 25, "loss")
    head(ax, 116, 68, 62, "⑤ 손실 함수 (학습 시)", "loss")
    t(ax, 147, 61.5, r"$\mathcal{L}=\mathrm{NLL}(T_{61})+w_s\,\mathrm{NLL}(\mathbf{s})+w_m\,\mathrm{NLL}(\log\dot m)$", 15,
      ha="center")
    t(ax, 147, 55.5, r"$+\ w_{phys}\,\Vert f(\mathbf{s}_{obs})-T_{61,obs}\Vert^2\ +\ w_{cons}\,\Vert\hat\mu+d-f(\hat{\mathbf{s}})\Vert^2$",
      15, ha="center")
    t(ax, 147, 50, "데이터 오차 + 물리 잔차 + 정합 잔차   ·   가중치 w 는 Optuna 로 탐색 (0.03~30)", 12.5, INK2,
      ha="center")

    # ⑥ 출력
    box(ax, 116, 76, 62, 14, "out")
    head(ax, 116, 86.5, 62, "⑥ 출력", "out")
    t(ax, 132, 80.5, r"$P(\mathrm{freeze})=\Phi\!\left(\dfrac{0-\hat\mu}{\hat\sigma}\right)$", 15, ha="center")
    t(ax, 164, 80.5, "→ 상위 1% 경보", 14, INK, ha="center", bold=True)
    box(ax, 156, 12, 22, 30, "out")
    head(ax, 156, 38, 22, "물리 분해", "out", 15.5)
    t(ax, 167, 30, "빙결 원인 설명", 12.5, INK2, ha="center")
    for i, s in enumerate(("지중 열교환 항", "헤더 온도 항", "줄-톰슨 냉각 항")):
        t(ax, 167, 24 - i * 4.6, s, 13, INK, ha="center")

    # 화살표 — 예측 경로(실선)
    arrow(ax, (32, 71), (40, 71))                                   # 데이터 → KF
    arrow(ax, (32, 52), (78, 52))                                   # 데이터 → 신경망
    arrow(ax, (70, 60), (78, 55), rad=0.0)                          # KF → 신경망
    arrow(ax, (93, 62), (93, 68))                                   # 은닉 → T61 헤드
    arrow(ax, (93, 42), (93, 36))                                   # 은닉 → 상태 헤드
    arrow(ax, (108, 24), (116, 24))                                 # 상태 헤드 → 물리층
    arrow(ax, (108, 80), (116, 80))                                 # T61 헤드 → 출력
    arrow(ax, (150, 27), (156, 27))                                 # 물리층 → 물리 분해
    # 학습 경로(점선)
    arrow(ax, (133, 42), (133, 47), color=C["loss"][0], dashed=True)            # 물리층 → 손실
    arrow(ax, (108, 71), (116, 66), color=C["loss"][0], dashed=True)            # T61 헤드 → 손실
    arrow(ax, (17, 12), (120, 12.3), color=C["phys"][0], dashed=True, rad=0.08)  # 현재 관측 → 물리층
    t(ax, 64, 3.2, "현재 관측 상태 → 물리 잔차 (유량이 관측된 시각만)", 12.5, C["phys"][0], ha="center")

    # 범례
    ax.add_patch(FancyArrowPatch((122, 3.2), (130, 3.2), arrowstyle="-|>", mutation_scale=18, color=INK2, lw=2.2))
    t(ax, 131.5, 3.2, "예측 경로", 12.5, INK2)
    ax.add_patch(FancyArrowPatch((148, 3.2), (156, 3.2), arrowstyle="-|>", mutation_scale=18, color=C["loss"][0], lw=2.2,
                                 linestyle=(0, (5, 4))))
    t(ax, 157.5, 3.2, "학습 시 손실 경로", 12.5, INK2)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor=SURF)
    plt.close(fig)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
