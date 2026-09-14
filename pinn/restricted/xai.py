"""XAI — 최종 제한 PINN 을 네 갈래로 설명한다.

    .venv/bin/python -m pinn.restricted.xai [--smoke]

1. **SHAP (GradientExplainer)** — 입력별 위험 기여. MC P(빙결)는 지시함수라 기울기가 없어
   `model.risk_score` = Φ((0 − T̂₆₁)/σ̂) (상태 평균을 물리층 통과, σ̂ 는 델타법)를 설명한다.
   배경 100 · 설명 200 (lstm/test 와 같은 규모). 빙결은 약 1% 로 드물어 무작위 200 행이면 사건이
   2개뿐이므로 **사건 최대 100 + 평시 100** 으로 층화한다.
2. **그룹 기여** — 현재계측 / 히터 / 유량 / 이력 / 계절·시각 / KF / AE / EKF.
   결측 플래그는 원래 열의 그룹에 합친다.
3. **물리 분해** (PINN 고유) — 6h 뒤 상태 평균으로
   T̂₆₁ = T_g(1−e^{−N}) + e^{−N}·T_hdr − e^{−N}·μ_JT·ΔP 의 세 항을 사건/평시로 비교.
4. **그룹 순열 중요도** (시험구간) — 그룹 열을 함께 섞어 AUC 하락. MC 난수는 평가마다 같은 시드
   (공통 난수)로 고정해 순열 효과만 남긴다. SHAP 의 교차검증.
+ Optuna 하이퍼파라미터 중요도·fold 별 물리 모수는 파이프라인이 저장한 것을 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from . import config, model as M

log = logging.getLogger("pinn.restricted.xai")
N_BG, N_EVENT, N_NORMAL, PERM_REPEATS, PERM_MC = 100, 100, 100, 5, 128

GROUPS = [
    ("KF", lambda c: c.startswith("kf_") or c in ("innov_t61", "innov_hdr", "innov_tin", "innov_t61_absmax")),
    ("AE", lambda c: c.startswith("ae_err")),
    ("EKF", lambda c: c.startswith("ekf_") or c in ("innov_ekf_t61", "nis_ekf_t61")),
    ("유량", lambda c: c in ("log_m", "m_obs")),
    ("히터", lambda c: c.startswith(("bath_", "out_", "duty_"))),
    ("이력", lambda c: c.startswith(("t61_mean", "t61_minr", "hdr_mean", "t61_trend", "t_in_mean"))),
    ("계절·시각", lambda c: c.startswith(("doy_", "hour_"))),
    ("현재계측", lambda c: True),
]


def group_of(name: str) -> str:
    base = name.removeprefix("결측_")
    return next(g for g, rule in GROUPS if rule(base))


class RiskWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.risk_score(x).unsqueeze(1)


def load(out_dir, train: str, dev: str):
    ck = torch.load(out_dir / f"model_{train}.pt", map_location=dev, weights_only=False)
    hp = ck["hp"]
    net = M.make(ck.get("arch", "hard"), ck["n_feat"], ck["ua_init_kw"], hp["hidden"], hp["n_layers"],
                 hp["dropout"], ck.get("dip", False)).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck, np.load(out_dir / f"xai_{train}.npz")


def explain(out_dir, train: str, dev: str) -> dict:
    net, ck, z = load(out_dir, train, dev)
    names = ck["names"]
    Zt, Zb, ev = z["Z_test"], z["Z_bg"], z["ev_test"].astype(int)
    rng = np.random.default_rng(config.RANDOM_SEED)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)

    # --- 1. SHAP (층화: 사건 최대 100 + 평시 100)
    i_ev = np.flatnonzero(ev == 1); i_no = np.flatnonzero(ev == 0)
    pick = np.concatenate([rng.choice(i_ev, min(N_EVENT, len(i_ev)), replace=False),
                           rng.choice(i_no, min(N_NORMAL, len(i_no)), replace=False)])
    bg = T(Zb[rng.choice(len(Zb), min(N_BG, len(Zb)), replace=False)])
    explainer = shap.GradientExplainer(RiskWrapper(net), bg)
    # cuDNN RNN 은 eval 모드 역전파를 거부한다 → SHAP 기울기 계산 동안만 cuDNN 을 끈다(2D 모델엔 무영향)
    with torch.backends.cudnn.flags(enabled=False):
        sv = explainer.shap_values(T(Zt[pick]))
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
    if sv.ndim >= 3 and sv.shape[-1] == 1:
        sv = sv[..., 0]
    if sv.ndim == 3:                             # LSTM (행, 창, 특징) → 창 방향 합 = 특징별 기여
        sv = sv.sum(axis=1)
    sv = sv.reshape(len(pick), -1)[:, :len(names)]   # LSTM 마스크 채널 제외
    is_ev = ev[pick] == 1
    feat = pd.DataFrame({"특징": names, "그룹": [group_of(n) for n in names],
                         "전체|SHAP|": np.abs(sv).mean(0), "사건|SHAP|": np.abs(sv[is_ev]).mean(0),
                         "평시|SHAP|": np.abs(sv[~is_ev]).mean(0), "사건평균SHAP": sv[is_ev].mean(0)})
    feat = feat.sort_values("전체|SHAP|", ascending=False)
    grp = feat.groupby("그룹")[["전체|SHAP|", "사건|SHAP|", "평시|SHAP|"]].sum()
    for c in grp.columns:
        grp[c.replace("|SHAP|", "_비중%")] = 100 * grp[c] / grp[c].sum()
    grp = grp.sort_values("전체|SHAP|", ascending=False)
    with torch.no_grad():
        r_pick = net.risk_score(T(Zt[pick])).cpu().numpy()
    np.savez_compressed(out_dir / f"shap_{train}.npz", values=sv, rows=pick, risk=r_pick,
                        is_event=is_ev, names=np.array(names))
    feat.to_csv(out_dir / f"shap_features_{train}.csv", index=False, encoding="utf-8-sig")
    grp.to_csv(out_dir / f"shap_groups_{train}.csv", encoding="utf-8-sig")

    # --- 3. 물리 분해 (6h 뒤 상태 평균)
    step = 4096 if Zt.ndim == 3 else len(Zt)                 # LSTM 입력은 조각으로 (GPU 메모리)
    with torch.no_grad():
        mu = torch.cat([net.heads(T(Zt[i:i + step]))[0] for i in range(0, len(Zt), step)])
        hdr, dp, tin, lm = mu.unbind(-1)
        parts = net.decompose(hdr, dp, tin, lm)
        t61_ce = (parts["지중"] + parts["헤더"] + parts["줄톰슨"]).cpu().numpy()
        dec = pd.DataFrame({k: v.cpu().numpy() for k, v in parts.items()} |
                           {"T61_확실성등가": t61_ce, "사건": ev})
    phys = dec.groupby("사건")[["지중", "헤더", "줄톰슨", "e", "T61_확실성등가"]].mean()
    phys.index = phys.index.map({0: "평시", 1: "사건"})
    phys.loc["사건−평시"] = phys.loc["사건"] - phys.loc["평시"]
    phys.to_csv(out_dir / f"physics_decomp_{train}.csv", encoding="utf-8-sig")

    # --- 4. 그룹 순열 중요도 (시험 전체, AUC 하락)
    groups = pd.Series([group_of(n) for n in names])

    def auc_of(Zm):
        torch.manual_seed(config.RANDOM_SEED)
        if dev == "cuda":
            torch.cuda.manual_seed_all(config.RANDOM_SEED)
        with torch.no_grad():
            p = np.concatenate([net.prob_freeze(T(Zm[i:i + step]), PERM_MC).cpu().numpy()
                                for i in range(0, len(Zm), step)])
        return roc_auc_score(ev, p) if 0 < ev.sum() < len(ev) else np.nan

    base_auc = auc_of(Zt)
    perm = []
    for g in groups.unique():
        cols_g = np.flatnonzero((groups == g).to_numpy())
        drops = []
        for r in range(PERM_REPEATS):
            Zp = Zt.copy()
            order = np.random.default_rng(1000 + r).permutation(len(Zt))
            Zp[..., cols_g] = Zt[order][..., cols_g]           # 그룹 열을 같은 순서로 함께 섞는다
            drops.append(base_auc - auc_of(Zp))
        perm.append({"그룹": g, "열수": len(cols_g), "AUC하락_평균": np.mean(drops),
                     "AUC하락_표준편차": np.std(drops, ddof=1)})
    perm = pd.DataFrame(perm).sort_values("AUC하락_평균", ascending=False)
    perm.insert(0, "기준AUC(보정전)", round(base_auc, 4))
    perm.to_csv(out_dir / f"perm_groups_{train}.csv", index=False, encoding="utf-8-sig")

    log.info("[%s] SHAP %d행(사건 %d) · 기준 AUC %.3f · 상위 그룹 %s · 순열 1위 %s",
             train, len(pick), int(is_ev.sum()), base_auc, grp.index[0], perm.iloc[0]["그룹"])
    return {"feat": feat, "grp": grp, "phys": phys, "perm": perm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trains", nargs="+", default=list(config.TRAINS))
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--arch", default="hard", choices=config.ARCHS)
    ap.add_argument("--target", default="min_raw", choices=config.TARGETS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = config.variant_dir(args.variant, args.smoke, args.arch, args.target)
    pd.set_option("display.width", 220)
    for t in args.trains:
        r = explain(out_dir, t, dev)
        print(f"\n=== {t} — SHAP 그룹 비중 ===")
        print(r["grp"].round(2).to_string())
        print(f"\n=== {t} — 상위 특징 ===")
        print(r["feat"].head(12).round(4).to_string(index=False))
        print(f"\n=== {t} — 물리 분해 (℃) ===")
        print(r["phys"].round(3).to_string())
        print(f"\n=== {t} — 그룹 순열 중요도 ===")
        print(r["perm"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
