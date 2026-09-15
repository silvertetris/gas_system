"""학습·평가. 문서 08 의 GBM 과 같은 분할(시험 2022~)에서 비교한다.

손실:
    NLL(T_61 최저, 미래)                         ← 주 목표
    w_state·NLL(헤더·압력강하·입구온도, 미래)     ← 전부 관측값
    w_m·NLL(log 유량, 미래)                      ← **유량 관측된 곳만**
    w_phys·동시점 물리 정합                       ← **유량 관측된 곳만** ("제한적")

## 구성 (2026-09-13 리팩터 — Optuna·XAI 용)

    prepare → tensors → fit → predict_*       ← 임의 분할 마스크·하이퍼파라미터를 받는다
    run(train)                                  ← 기존 진입점. 난수 호출 순서를 그대로 유지해
                                                  이전 결과(시드 42, 정제 없음 → M AUC 0.849)를 재현한다
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

DEFAULT_HP = {"hidden": config.HIDDEN, "n_layers": 2, "dropout": 0.0, "lr": config.LR,
              "batch": config.BATCH, "weight_decay": 0.0, "w_state": config.W_STATE,
              "w_m": config.W_M, "w_phys": config.W_PHYS}


def metrics(p: np.ndarray, ev: np.ndarray, base: float) -> dict:
    # ⚠ 등장성 보정 출력이 부동소수 반올림으로 1.0000001 이 될 수 있다 — sklearn 이 거부한다
    #   (시드 123 ＋AE Z 에서 실제로 중단됐다). 확률이므로 [0,1] 로 자른다. 값은 바뀌지 않는다.
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    if ev.sum() == 0 or ev.sum() == len(ev):
        return {"AUC": np.nan, "Brier개선%": np.nan, "리프트1": np.nan, "리프트5": np.nan,
                "사건수": int(ev.sum())}
    r = {"AUC": round(float(roc_auc_score(ev, p)), 3),
         "Brier개선%": round(100 * (1 - brier_score_loss(ev, p)
                                 / brier_score_loss(ev, np.full_like(p, base))), 1),
         "사건수": int(ev.sum())}
    for frac in (0.01, 0.05):
        k = max(int(frac * len(p)), 10)
        top = np.argsort(-p)[:k]
        r[f"리프트{int(frac * 100)}"] = round(float(ev[top].mean() / ev.mean()), 1)
    return r


def prepare(X: pd.DataFrame, cols: list[str], tr: np.ndarray):
    """표준화 + 결측 0 대치 + 결측 플래그. 통계는 **학습 마스크로만** 계산한다."""
    A = X[cols].to_numpy(np.float64)
    mu_, sd_ = np.nanmean(A[tr], 0), np.nanstd(A[tr], 0)
    sd_[~np.isfinite(sd_) | (sd_ < 1e-9)] = 1.0
    mu_[~np.isfinite(mu_)] = 0.0
    Zs = (A - mu_) / sd_
    miss = ~np.isfinite(Zs)
    flag_cols = [i for i in range(A.shape[1]) if miss[:, i].any()]
    Zs = np.where(miss, 0.0, Zs)
    Zs = np.hstack([Zs, miss[:, flag_cols].astype(float)]).astype(np.float32)
    names = list(cols) + [f"결측_{cols[i]}" for i in flag_cols]
    return Zs, names


def sequence(X: pd.DataFrame, Zs: np.ndarray, window: int) -> np.ndarray:
    """(행, 특징) → (행, 창, 특징+1). 시각 t 의 창 = [t−window+1, …, t] (과거만, 누수 없음).

    창 안의 시각이 X 에 없으면(결측·필터로 빠진 행) 0 벡터 + 마스크 0. 마지막 채널이 행 존재 마스크.
    """
    step = pd.Timedelta(config.FREQ)
    h = np.asarray((X.index - X.index[0]) // step, dtype=np.int64)
    pos = np.full(int(h.max()) + 1, -1, dtype=np.int64)
    pos[h] = np.arange(len(X))
    lag = h[:, None] - np.arange(window - 1, -1, -1)[None, :]
    ok = lag >= 0
    idx = np.where(ok, pos[np.clip(lag, 0, None)], -1)
    pad = np.vstack([Zs, np.zeros((1, Zs.shape[1]), np.float32)])
    out = pad[idx]                                                   # idx −1 → 마지막(0) 행
    mask = (idx >= 0).astype(np.float32)[..., None]
    return np.concatenate([out, mask], axis=2).astype(np.float32)


def tensors(X: pd.DataFrame, cols: list[str], tr: np.ndarray, dev: str, window: int | None = None):
    Zs, names = prepare(X, cols, tr)
    if window:
        Zs = sequence(X, Zs, int(window))

    def T(a):
        return torch.tensor(np.asarray(a, dtype=np.float32), device=dev)

    y = X["y_t61_min"].to_numpy(np.float32)
    ymean = (X["y_t61_mean"] if "y_t61_mean" in X else X["y_t61_min"]).to_numpy(np.float32)
    wmean = np.isfinite(ymean).astype(np.float32)
    ymean = np.nan_to_num(ymean)
    S = X[["y_hdr", "y_dp", "y_tin"]].to_numpy(np.float32)
    ym = X["y_log_m"].to_numpy(np.float32)
    wm = np.isfinite(ym).astype(np.float32)
    ym = np.nan_to_num(ym)
    now = X[["hdr", "dp", "t_in", "m_tot", "t61"]].to_numpy(np.float32)
    wp = (X["m_obs"].to_numpy() > 0.5).astype(np.float32) * np.isfinite(now).all(1)
    now = np.nan_to_num(now, nan=1.0)
    var_t61 = float(np.nanvar(X["t61"].to_numpy()[tr]))
    D = {k: T(v) for k, v in dict(Z=Zs, y=y, S=S, ym=ym, wm=wm, now=now, wp=wp, ymean=ymean, wmean=wmean).items()}
    return D, var_t61, names, y, wp


def loss_fn(net, D, idx, var_t61: float, hp: dict, n_mc: int = config.N_MC):
    if isinstance(net, M.SoftPINN):
        return _loss_soft(net, D, idx, var_t61, hp)
    mu, sd, smu, ssd, _ = net(D["Z"][idx], n_mc)
    l = M.nll(mu, sd, D["y"][idx])
    for j in range(3):
        l = l + hp["w_state"] * M.nll(smu[:, j], ssd[:, j], D["S"][idx, j])
    l = l + hp["w_m"] * M.nll(smu[:, 3], ssd[:, 3], D["ym"][idx], D["wm"][idx])
    n = D["now"][idx]
    phys = net.supply(n[:, 0], n[:, 1], n[:, 2], n[:, 3])
    w = D["wp"][idx]
    return l + hp["w_phys"] * (((phys - n[:, 4]) ** 2 / var_t61) * w).sum() / w.sum().clamp_min(1.0)


def _loss_soft(net, D, idx, var_t61: float, hp: dict):
    """soft·nn — T61 직접 NLL + 상태 NLL (+ soft 만: 동시점 물리 잔차 + 정합 잔차 + 순간하강 지도)."""
    mu, sd, smu, ssd, d = net.outputs(D["Z"][idx])
    l = M.nll(mu, sd, D["y"][idx])
    for j in range(3):
        l = l + hp["w_state"] * M.nll(smu[:, j], ssd[:, j], D["S"][idx, j])
    l = l + hp["w_m"] * M.nll(smu[:, 3], ssd[:, 3], D["ym"][idx], D["wm"][idx])
    if not net.use_physics:
        return l
    n = D["now"][idx]
    phys = net.supply(n[:, 0], n[:, 1], n[:, 2], n[:, 3])
    w = D["wp"][idx]
    l = l + hp["w_phys"] * (((phys - n[:, 4]) ** 2 / var_t61) * w).sum() / w.sum().clamp_min(1.0)
    f_s = net.supply_log(smu[:, 0], smu[:, 1], smu[:, 2], smu[:, 3])
    l = l + hp.get("w_cons", 1.0) * ((mu + d - f_s) ** 2 / var_t61).mean()
    if net.dip:
        wm_ = D["wmean"][idx]
        l = l + (((mu + d - D["ymean"][idx]) ** 2 / var_t61) * wm_).sum() / wm_.sum().clamp_min(1.0)
    return l


def _accumulate(net, D, bidx, var_t61: float, hp: dict, n_mc: int, parts: int = 4):
    """GPU 메모리 부족 시에만: 배치를 조각내 기울기를 누적한다. 조각 손실을 크기 비율로 가중해 평균 손실과 같은 방향."""
    total = torch.zeros((), device=bidx.device)
    for part in bidx.chunk(parts):
        l = loss_fn(net, D, part, var_t61, hp, n_mc)
        if not torch.isfinite(l):
            return l.detach()
        w = len(part) / len(bidx)
        (l * w).backward()
        total = total + l.detach() * w
    return total


def fit(train: str, D: dict, var_t61: float, tr: np.ndarray, va: np.ndarray, hp: dict | None = None,
        epochs: int = config.EPOCHS, patience: int = config.PATIENCE, n_mc: int = config.N_MC,
        dev: str = "cuda", log_every: int | None = 25, epoch_cb=None, history: list | None = None,
        arch: str = "hard", dip: bool = False):
    """조기종료 학습. `epoch_cb(ep, valid_loss)` 가 True 를 돌려주면 중단(Optuna 가지치기용).
    `history` 를 주면 epoch 마다 학습·검증 손실과 물리 모수를 쌓는다(학습곡선 그림용, 난수 소비 없음)."""
    hp = {**DEFAULT_HP, **(hp or {})}
    net = M.make(arch, D["Z"].shape[-1], config.UA_PIPE_INIT_KW[train],
                 hp["hidden"], hp["n_layers"], hp["dropout"], dip).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    itr = torch.tensor(np.flatnonzero(tr), device=dev)
    iva = torch.tensor(np.flatnonzero(va), device=dev)
    best, best_state, bad, n_skip, n_oom = np.inf, None, 0, 0, 0
    for ep in range(epochs):
        net.train()
        perm = itr[torch.randperm(len(itr), device=dev)]
        tl, tn = torch.zeros((), device=dev), 0
        for i in range(0, len(perm), hp["batch"]):
            opt.zero_grad()
            bidx = perm[i:i + hp["batch"]]
            oom = False
            try:
                loss = loss_fn(net, D, bidx, var_t61, hp, n_mc)
                if torch.isfinite(loss):
                    loss.backward()
            except torch.OutOfMemoryError:
                oom = True
            if oom:
                # 여러 작업이 GPU 를 나눠 쓸 때만 발생(2026-09-14 LSTM 창 48·배치 4096). 평소 경로는 그대로다.
                loss = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                n_oom += 1
                loss = _accumulate(net, D, bidx, var_t61, hp, n_mc)
            # 안전장치: 손실·기울기가 유한하지 않은 배치는 갱신하지 않는다(한 배치로 가중치 전체가 NaN 이 되는 것 방지).
            #   유한한 배치에서는 계산이 그대로라 결과가 바뀌지 않는다. 건너뛴 수는 끝에 경고로 남긴다.
            if not torch.isfinite(loss):
                n_skip += 1
                continue
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)):
                n_skip += 1
                continue
            opt.step()
            tl, tn = tl + loss.detach(), tn + 1
        net.eval()
        with torch.no_grad():
            if D["Z"].ndim == 3 and len(iva) > SEQ_CHUNK:        # LSTM: 한 번에 넣으면 cuDNN 작업공간이 수 GB
                v = sum(float(loss_fn(net, D, iva[i:i + SEQ_CHUNK], var_t61, hp, n_mc)) * len(iva[i:i + SEQ_CHUNK])
                        for i in range(0, len(iva), SEQ_CHUNK)) / len(iva)
            else:
                v = float(loss_fn(net, D, iva, var_t61, hp, n_mc))
        if history is not None:
            history.append({"epoch": ep, "train": float(tl) / max(tn, 1), "valid": v,
                            "μ_JT": net.mu_jt.item(), "UA_kW/K": net.ua_kw.item(), "N₀": net.n0.item()})
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
        if log_every and ep % log_every == 0:
            log.info("  ep %3d valid %.4f | μ_JT %.3f  UA %.2f kW/K  N₀ %.3f  T_g=%.2f%+.3f·T_in",
                     ep, v, net.mu_jt.item(), net.ua_kw.item(), net.n0.item(),
                     net.tg_a.item(), net.tg_b.item())
        if epoch_cb is not None and epoch_cb(ep, v):
            break
    if n_skip:
        log.warning("  비유한 배치 %d개 건너뜀 (epoch %d 까지)", n_skip, ep)
    if n_oom:
        log.warning("  GPU 메모리 부족 배치 %d개를 조각 누적으로 처리", n_oom)
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, best


CHUNK_ROWS = 50_000         # 이보다 크면 조각으로 나눠 MC 평가 (GPU 8GB 보호)
SEQ_CHUNK = 4_096           # 시계열(LSTM) 입력의 조각 크기 — 2D 입력(hard·soft·nn)에는 적용 안 함(결과 재현 유지)


@torch.no_grad()
def predict_prob(net, D, idx: np.ndarray, dev: str, n_mc: int = 256) -> np.ndarray:
    """MC P(빙결). 행이 CHUNK_ROWS 이하면 한 번에(이전과 같은 난수 순서 → 1시간 결과 재현),
    넘으면 조각으로 나눈다(10분 해상도 시험구간 약 12.7만 × 표본 256 은 한 번에 올리면 위험)."""
    pos = np.flatnonzero(idx)
    chunk = SEQ_CHUNK if D["Z"].ndim == 3 else CHUNK_ROWS
    if len(pos) <= chunk:
        return net.prob_freeze(D["Z"][torch.tensor(pos, device=dev)], n_mc).cpu().numpy()
    return np.concatenate([net.prob_freeze(D["Z"][torch.tensor(pos[i:i + chunk], device=dev)], n_mc).cpu().numpy()
                           for i in range(0, len(pos), chunk)])


@torch.no_grad()
def predict_mean_sd(net, D, ite: torch.Tensor):
    """시험구간 T61 예측 평균·표준편차. 시계열 입력이면 조각으로."""
    if D["Z"].ndim != 3:
        mu, sd = net(D["Z"][ite])[:2]
        return mu.cpu().numpy(), sd.cpu().numpy()
    parts = [net(D["Z"][ite[i:i + SEQ_CHUNK]])[:2] for i in range(0, len(ite), SEQ_CHUNK)]
    return (torch.cat([m for m, _ in parts]).cpu().numpy(), torch.cat([s for _, s in parts]).cpu().numpy())


def run(train: str) -> dict:
    """기존 진입점 — 문서 16~18 분할(시험 2022~). 난수 호출 순서를 이전 판과 같게 유지한다."""
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    f = data.build(train)
    X, cols, tr, va, te = data.split(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("[%s] 표본 %d | 학습 %d 검증 %d 시험 %d | 특징 %d | 유량관측 %.1f%%",
             train, len(X), tr.sum(), va.sum(), te.sum(), len(cols),
             100 * X["m_obs"].mean())

    D, var_t61, _, y, wp = tensors(X, cols, tr, dev)
    ev = (y < config.FREEZE_C).astype(float)
    net, _ = fit(train, D, var_t61, tr, va, DEFAULT_HP, dev=dev)

    with torch.no_grad():
        iva = torch.tensor(np.flatnonzero(va), device=dev)
        p_va = net.prob_freeze(D["Z"][iva]).cpu().numpy()
        ite = torch.tensor(np.flatnonzero(te), device=dev)
        p_te = net.prob_freeze(D["Z"][ite]).cpu().numpy()
        mu_te = net(D["Z"][ite])[0].cpu().numpy()
        n = D["now"][ite]
        phys_now = net.supply(n[:, 0], n[:, 1], n[:, 2], n[:, 3]).cpu().numpy()
        N_now = net.n_pipe(n[:, 3]).cpu().numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_va, ev[va])
    p_cal = iso.predict(p_te)
    base = float(ev[tr].mean())

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6,
                                         l2_regularization=1.0, random_state=config.RANDOM_SEED,
                                         early_stopping=True, validation_fraction=0.15)
    clf.fit(X.loc[tr, cols], ev[tr])
    iso_g = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        clf.predict_proba(X.loc[va, cols])[:, 1], ev[va])
    g_cal = iso_g.predict(clf.predict_proba(X.loc[te, cols])[:, 1])

    obs_te = X["m_obs"].to_numpy()[te] > 0.5
    ev_te = ev[te]
    out = {"계열": train, "시험n": int(te.sum()), "시험사건률": round(float(ev_te.mean()), 4),
           "유량관측%": round(100 * float(obs_te.mean()), 1),
           "μ_JT": round(net.mu_jt.item(), 4), "UA_배관_kW/K": round(net.ua_kw.item(), 2),
           "N₀": round(net.n0.item(), 3),
           "T_g": f"{net.tg_a.item():+.2f}{net.tg_b.item():+.3f}·T_in",
           "헤더보존율exp(−N)중앙": round(float(np.median(np.exp(-N_now[obs_te]))), 3)
           if obs_te.any() else np.nan}
    wp_te = wp[te] > 0.5
    t61_te = X["t61"].to_numpy()[te]
    out["물리정합RMSE_유량관측"] = round(float(np.sqrt(np.mean((phys_now - t61_te)[wp_te] ** 2))), 2)
    out["공급온도예측RMSE"] = round(float(np.sqrt(np.mean((mu_te - y[te]) ** 2))), 2)
    for tag, mask in (("전체", np.ones_like(obs_te)), ("유량관측", obs_te), ("유량미관측", ~obs_te)):
        for name, p in (("PINN", p_cal), ("GBM", g_cal)):
            for k, v in metrics(p[mask], ev_te[mask], base).items():
                out[f"{name}_{tag}_{k}"] = v

    # --- 유량 구간별 물리 성립 점검 (시험, 유량 관측분)
    d = pd.DataFrame({"m": n[:, 3].cpu().numpy(), "t61": t61_te, "hdr": n[:, 0].cpu().numpy(),
                      "dp": n[:, 1].cpu().numpy(), "tin": n[:, 2].cpu().numpy(),
                      "k_model": np.exp(-N_now)})[obs_te]
    rows = []
    if len(d) > 600:
        qs = np.quantile(d["m"], np.linspace(0, 1, 6))
        for lo, hi in zip(qs[:-1], qs[1:]):
            b = d[(d["m"] >= lo) & (d["m"] <= hi)]
            if len(b) < 100:
                continue
            treg = b["hdr"] - net.mu_jt.item() * b["dp"]
            Xb = np.column_stack([np.ones(len(b)), treg, b["tin"]])
            c, *_ = np.linalg.lstsq(Xb, b["t61"].to_numpy(), rcond=None)
            rows.append({"계열": train, "유량구간": f"{lo:.2f}~{hi:.2f}", "n": len(b),
                         "k_경험(회귀)": round(float(c[1]), 3),
                         "k_모델 exp(−N)": round(float(b["k_model"].median()), 3)})
    pred = pd.DataFrame({"p": p_cal, "p_gbm": g_cal, "ev": ev_te, "mu": mu_te,
                         "t61_now": t61_te, "phys_now": phys_now, "N": N_now,
                         "m_obs": obs_te}, index=X.index[te])
    return out, pd.DataFrame(rows), pred


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for t in config.TRAINS:
        out, chk, pred = run(t)
        rows.append(out)
        checks.append(chk)
        pred.to_parquet(config.OUTPUT_DIR / f"pred_{t}.parquet")
    m = pd.DataFrame(rows)
    c = pd.concat(checks, ignore_index=True)
    m.to_csv(config.OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    c.to_csv(config.OUTPUT_DIR / "flowcheck.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    base = ["계열", "시험n", "시험사건률", "유량관측%"]
    print("\n=== 식별된 물리 모수 ===")
    print(m[base[:1] + ["μ_JT", "UA_배관_kW/K", "N₀", "T_g", "헤더보존율exp(−N)중앙",
                        "물리정합RMSE_유량관측", "공급온도예측RMSE"]].to_string(index=False))
    for tag in ("전체", "유량관측", "유량미관측"):
        k = [x for x in m.columns if f"_{tag}_" in x]
        print(f"\n=== 빙결 예측 — {tag} ===")
        print(m[base + k].to_string(index=False))
    print("\n=== 유량 구간별: 경험 보존율 vs 모델 exp(−N) ===")
    print(c.to_string(index=False))


if __name__ == "__main__":
    main()
