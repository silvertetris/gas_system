"""학습·평가. 기준선(GBM)과 같은 시험구간에서 비교한다.

## 손실 구성

    예측    NLL(T_61(t+h))                              ← 주 목표
    상태    NLL(T_in, ΔP, valve, T_bath,u, ε_u at t+h)   ← **전부 관측 가능** → 전부 지도
    헤더    (Q4) 물리층 출력을 y_hdr_min 에 맞춘다        ← 자유 헤드가 아니다
    (Q4)    동시점 T_hdr 잔차                            ← β(밸브) 단조맵을 묶는다
    (Q5·Q6) 동시점 T_61  잔차                            ← μ_JT·T_g·N 을 묶는다

⚠ (Q3) 잔차는 손실에 없다 — **항등적으로 0** 이다. ε 을 실측으로 쓰면
  `T_out = T_in + ε·(T_bath − T_in)` 이 ε 의 정의와 같다.

동시점 잔차는 미래값을 쓰지 않으므로 누수가 없다. 미래 예측으로의 정보 전달은
**공유 물리 모수**(μ_JT·T_g·β 단조맵)를 통해서만 일어난다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config, data, model as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _metrics(p: np.ndarray, ev: np.ndarray, base: float) -> dict:
    r = {"AUC": round(float(roc_auc_score(ev, p)), 3),
         "Brier": round(float(brier_score_loss(ev, p)), 5),
         "Brier개선%": round(100 * (1 - brier_score_loss(ev, p)
                                 / brier_score_loss(ev, np.full_like(p, base))), 1),
         "평균예측": round(float(p.mean()), 4)}
    for frac in (0.01, 0.05):
        k = max(int(frac * len(p)), 10)
        top = np.argsort(-p)[:k]
        r[f"리프트{int(frac*100)}"] = (round(float(ev[top].mean() / ev.mean()), 1)
                                     if ev.mean() > 0 else np.nan)
    return r


def run(train: str) -> dict:
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    units = config.TRAINS[train]["units"]
    f = data.build(train)
    X, cols, tr, va, te = data.split(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mu_ = X.loc[tr, cols].to_numpy(dtype=np.float64).mean(axis=0)
    sd_ = X.loc[tr, cols].to_numpy(dtype=np.float64).std(axis=0)
    sd_[sd_ < 1e-9] = 1.0
    Z = ((X[cols].to_numpy(dtype=np.float64) - mu_) / sd_).astype(np.float32)

    def T(a):
        return torch.tensor(np.asarray(a, dtype=np.float32), device=dev)

    def col(name):
        return X[name].to_numpy(dtype=np.float32)

    y = col("y_t61_min")
    # 상태 순서는 model.FreezePINN 주석과 같아야 한다: [t_in, dp, bath_u...]
    state_cols = (["y_tin", "y_dp_max", "y_valve"] + [f"y_bath_{u}" for u in units]
                  + [f"y_eps_{u}" for u in units])
    y_state = X[state_cols].to_numpy(dtype=np.float32)
    y_hdr = col("y_hdr_min")
    # 동시점 실측 (물리 잔차용)
    o_bath = X[[f"bath_{u}" for u in units]].to_numpy(dtype=np.float32)
    o_eps = X[[f"eps_{u}" for u in units]].to_numpy(dtype=np.float32)
    o_tin, o_dp, o_hdr, o_t61 = col("t_in"), col("dp"), col("hdr"), col("t61")
    o_valve = col("valve")
    ev = (y < config.FREEZE_C).astype(float)
    o_beta_obs = X["beta"].to_numpy(dtype=np.float64)   # 로그 표시용 (손실에는 쓰지 않는다)
    # ε 상태 지도의 가중 — 구동온도차가 작으면 실측 ε 이 의미 없으므로 지도하지 않는다.
    # 물리층의 `avail` 과 **같은 식**을 쓴다 (model.heater 주석 참고).
    y_drive = X[[f"y_drive_{u}" for u in units]].to_numpy(dtype=np.float32)
    s_w = np.ones((len(X), 3 + 2 * len(units)), dtype=np.float32)
    s_w[:, 3 + len(units):] = 1.0 / (1.0 + np.exp(
        -(y_drive - config.MIN_DRIVE_C) / config.DRIVE_SOFT_C))

    knots = np.unique(np.quantile(o_valve[tr], np.linspace(0.05, 0.95, config.BETA_KNOTS)))
    net = M.FreezePINN(len(cols), units, knots).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=config.LR)

    # --- 잔차 정규화 스케일 (훈련구간 분산). 없으면 물리항이 손실을 지배한다(4차 버그).
    def var_tr(a):
        v = float(np.nanvar(np.asarray(a, dtype=np.float64)[tr]))
        return max(v, 1e-6)

    v_hdr, v_t61, v_yhdr = var_tr(o_hdr), var_tr(o_t61), var_tr(y_hdr)

    def pack(idx):
        return {k: T(v[idx]) for k, v in
                dict(Z=Z, y=y, S=y_state, SW=s_w, hdr=y_hdr, bath=o_bath, eps=o_eps,
                     tin=o_tin, dp=o_dp, valve=o_valve, ohdr=o_hdr, t61=o_t61).items()}

    D_tr, D_va = pack(tr), pack(va)

    def phys_loss(d, b=None):
        """동시점 (Q4)·(Q5)·(Q6) 잔차. 실측 상태를 그대로 물리층에 넣는다."""
        sl = slice(None) if b is None else b
        n = net.now(d["Z"][sl], d["eps"][sl], d["bath"][sl], d["tin"][sl],
                    d["dp"][sl], d["valve"][sl])
        l4 = (((n["t_hdr"] - d["ohdr"][sl]) ** 2) / v_hdr).mean()
        l6 = (((n["t61"] - d["t61"][sl]) ** 2) / v_t61).mean()
        return config.W_Q4 * l4 + config.W_PHYS * l6, n

    def total_loss(d, b=None):
        sl = slice(None) if b is None else b
        mu, sd, parts = net(d["Z"][sl])
        loss = M.nll(mu, sd, d["y"][sl])
        for j in range(net.n_state):
            w = d["SW"][sl, j]
            nl = (torch.log(parts["sd"][:, j])
                  + 0.5 * ((d["S"][sl, j] - parts["mu"][:, j]) / parts["sd"][:, j]) ** 2)
            loss = loss + config.W_STATE * (nl * w).sum() / w.sum().clamp_min(1.0)
        # 물리층이 낸 미래 헤더온도가 관측을 맞혀야 한다 (자유 헤드가 아니다)
        loss = loss + config.W_STATE * (
            ((parts["t_hdr"].mean(0) - d["hdr"][sl]) ** 2).mean() / v_yhdr)
        p, _ = phys_loss(d, b)
        return loss + p

    best, best_state, bad = np.inf, None, 0
    n_tr = len(tr.nonzero()[0])
    for ep in range(config.EPOCHS):
        net.train()
        perm = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr, config.BATCH):
            b = perm[i:i + config.BATCH]
            opt.zero_grad()
            total_loss(D_tr, b).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(total_loss(D_va))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad > config.PATIENCE:
                break
        if ep % 50 == 0:
            with torch.no_grad():
                nw = net.now(D_va["Z"], D_va["eps"], D_va["bath"], D_va["tin"],
                             D_va["dp"], D_va["valve"])
            log.info("  ep %4d valid %.4f | μ_JT %.3f T_g=%.2f%+.3f·T_in | "
                     "β중앙 %.3f (실측 %.3f) | 가중 %s",
                     ep, v, float(net.mu_jt), float(net.tg_a.detach()), float(net.tg_b),
                     float(nw["beta"].median()), float(np.nanmedian(o_beta_obs[va])),
                     np.round(nw["f"].median(0).values.cpu().numpy(), 3))
    if best_state:
        net.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        D_te = pack(te)
        p_va = net.prob_freeze(D_va["Z"]).cpu().numpy()
        p_te = net.prob_freeze(D_te["Z"]).cpu().numpy()
        mu_te, sd_te, parts = net(D_te["Z"])
        mu_te, sd_te = mu_te.cpu().numpy(), sd_te.cpu().numpy()
        st = parts["mu"].cpu().numpy()
        hdr_te = parts["t_hdr"].mean(0).cpu().numpy()
        f_te = parts["f"].mean(0).cpu().numpy()
        eps_te = parts["eps"].mean(0).cpu().numpy()
        beta_te = parts["beta"].mean(0).cpu().numpy()
        n_pipe = parts["n_pipe"].cpu().numpy()
        nw_te = net.now(D_te["Z"], D_te["eps"], D_te["bath"], D_te["tin"],
                        D_te["dp"], D_te["valve"])
    base = float(ev[tr].mean())

    out = {"계열": train, "μ_JT": round(float(net.mu_jt), 4),
           "T_g": f"{float(net.tg_a):+.2f}{float(net.tg_b):+.3f}·T_in",
           "N_배관중앙": round(float(np.median(n_pipe)), 3),
           "헤더보존율": round(float(np.median(np.exp(-n_pipe))), 3),
           "훈련사건률": round(base, 4), "시험사건률": round(float(ev[te].mean()), 4),
           "시험n": int(te.sum())}
    for k, u in enumerate(units):
        out[f"ε예측중앙_{u}"] = round(float(np.median(eps_te[:, k])), 3)
        out[f"ε실측중앙_{u}"] = round(float(X.loc[te, f"eps_{u}"].median()), 3)
        out[f"가중중앙_{u}"] = round(float(np.median(f_te[:, k])), 3)
    out["β중앙"] = round(float(np.median(beta_te)), 3)
    out["β실측중앙"] = round(float(X.loc[te, "beta"].median()), 3)
    out.update({f"PINN_{k}": v for k, v in _metrics(p_te, ev[te], base).items()})
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_va, ev[va])
    out.update({f"PINN보정_{k}": v for k, v in _metrics(iso.predict(p_te), ev[te], base).items()})

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        random_state=config.RANDOM_SEED, early_stopping=True, validation_fraction=0.15)
    clf.fit(X.loc[tr, cols], ev[tr])
    g_va, g_te = clf.predict_proba(X.loc[va, cols])[:, 1], clf.predict_proba(X.loc[te, cols])[:, 1]
    iso2 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso2.fit(g_va, ev[va])
    out.update({f"GBM_{k}": v for k, v in _metrics(iso2.predict(g_te), ev[te], base).items()})

    # --- 물리 정합 (동시점, 시험구간). (Q3) 는 항등이라 검사 대상이 아니다.
    out["Q4_헤더RMSE"] = round(float(np.sqrt(np.mean(
        (nw_te["t_hdr"].cpu().numpy() - X.loc[te, "hdr"].to_numpy()) ** 2))), 2)
    out["Q6_공급RMSE"] = round(float(np.sqrt(np.mean(
        (nw_te["t61"].cpu().numpy() - X.loc[te, "t61"].to_numpy()) ** 2))), 2)
    out["미래헤더RMSE"] = round(float(np.sqrt(np.mean((hdr_te - y_hdr[te]) ** 2))), 2)
    out["미래공급RMSE"] = round(float(np.sqrt(np.mean((mu_te - y[te]) ** 2))), 2)
    for j, c in enumerate(state_cols):
        out[f"상태RMSE_{c}"] = round(float(np.sqrt(np.mean((st[:, j] - y_state[te, j]) ** 2))), 2)

    pred = {"p": p_te, "ev": ev[te], "mu": mu_te, "sd": sd_te, "mu_hdr": hdr_te,
            "mu_tin": st[:, 0], "mu_dp": st[:, 1], "mu_valve": st[:, 2],
            "beta": beta_te, "N": n_pipe,
            "t_g": float(net.tg_a) + float(net.tg_b) * st[:, 0]}
    for k, u in enumerate(units):
        pred[f"f_{u}"], pred[f"eps_{u}"] = f_te[:, k], eps_te[:, k]
    df_pred = pd.DataFrame(pred, index=X.index[te])
    for k, u in enumerate(units):
        df_pred[f"out_obs_{u}"] = X.loc[te, f"out_{u}"].to_numpy()
        df_pred[f"out_now_{u}"] = nw_te["t_out"][:, k].cpu().numpy()
        df_pred[f"eps_obs_{u}"] = X.loc[te, f"eps_{u}"].to_numpy()
    df_pred["hdr_obs"] = X.loc[te, "hdr"].to_numpy()
    df_pred["hdr_now"] = nw_te["t_hdr"].cpu().numpy()
    df_pred["beta_obs"] = X.loc[te, "beta"].to_numpy()
    df_pred.attrs["mu_jt"] = float(net.mu_jt)
    out["_pred"] = df_pred
    return out


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in config.TRAINS:
        log.info("=== 계열 %s ===", t)
        r = run(t)
        r.pop("_pred").to_csv(config.OUTPUT_DIR / f"pred_{t}.csv")
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(config.OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for grp, keys in (("물리 모수", ["계열", "μ_JT", "T_g", "헤더보존율"]),
                      ("히터 블록 (ε 관측)", [c for c in out.columns
                                 if c.split("_")[0] in {"ε예측중앙", "ε실측중앙", "가중중앙"}]),
                      ("합류점", ["β중앙", "β실측중앙"]),
                      ("물리 정합 RMSE", [c for c in out.columns if "RMSE" in c]),
                      ("예측 성능", [c for c in out.columns
                                 if c.startswith(("PINN보정_", "GBM_")) or c.endswith("사건률")])):
        k = [c for c in keys if c in out.columns]
        if k:
            print(f"\n=== {grp} ===")
            print(out[(["계열"] if "계열" not in k else []) + k].to_string(index=False))


if __name__ == "__main__":
    main()
