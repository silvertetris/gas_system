"""검정 2 — 시불변 상수 제약 하에서 합류점을 맞히는가. 대조군 동반.

학습 ~2021, 퍼지 1년, 홀드아웃 2022~. `ua_log` 와 β(밸브) 는 **학습구간에서만** 적합하고
홀드아웃에서 평가한다. 가중방식만 바꾼 대조군 7개를 같은 조건에서 돌린다.

읽는 법:
  · `ntu_series` 가 `equal`·`design` 보다 뚜렷이 낫다 → 열전달식이 유량 배분을 설명한다
  · `ntu_series` ≈ `equal` → 유량 가중이 기여하지 않는다. 식이 합류점을 설명하지 못한다
  · `ntu_series` ≈ `free` → 물리 제약의 손해가 없다
  · `ntu_series` ≈ `eps_shuffle` → ε 의 **시간 정보**가 쓰이지 않는다. 치명적
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from . import config, data, model as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _fit(w: str, train: str, d: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
         dev: str) -> dict:
    torch.manual_seed(config.RANDOM_SEED)
    units = config.TRAINS[train]["units"]
    cols = data.feature_cols(train)

    def T(a):
        return torch.tensor(np.asarray(a, dtype=np.float32), device=dev)

    F = d[cols].to_numpy(dtype=np.float32)
    mu, sd = F[tr].mean(0), F[tr].std(0)
    sd[sd < 1e-9] = 1.0
    F = (F - mu) / sd

    eps = d[[f"eps_{u}" for u in units]].to_numpy(dtype=np.float32)
    if w.startswith("eps_shuffle"):
        rng = np.random.default_rng(config.RANDOM_SEED)
        eps = eps[rng.permutation(len(eps))]          # 시간 정보 파괴 (영가설)
    out = d[[f"out_{u}" for u in units]].to_numpy(dtype=np.float32)
    tin = d["t_in"].to_numpy(dtype=np.float32)
    val = d["valve"].to_numpy(dtype=np.float32)
    hdr = d["hdr"].to_numpy(dtype=np.float32)
    gate = (d[[f"regime_{u}" for u in units]] == "isolated").to_numpy(dtype=np.float32)
    fgate = (d[[f"regime_{u}" for u in units]] == "flow").to_numpy(dtype=np.float32)
    duty = d[[f"duty_{u}" for u in units]].to_numpy(dtype=np.float32)
    dsin = d["doy_sin"].to_numpy(dtype=np.float32)
    dcos = d["doy_cos"].to_numpy(dtype=np.float32)

    net = M.MixingTest(w, units, len(cols), M.valve_knots(val[tr])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=config.LR)

    itr, ite = np.flatnonzero(tr), np.flatnonzero(te)
    Ftr, Etr, Otr, Ttr, Vtr, Htr = (T(F[itr]), T(eps[itr]), T(out[itr]),
                                    T(tin[itr]), T(val[itr]), T(hdr[itr]))
    Gtr, FGtr, Dtr = T(gate[itr]), T(fgate[itr]), T(duty[itr])
    Str, Ctr = T(dsin[itr]), T(dcos[itr])
    for ep in range(config.EPOCHS):
        perm = torch.randperm(len(itr), device=dev)
        for i in range(0, len(itr), config.BATCH):
            b = perm[i:i + config.BATCH]
            opt.zero_grad()
            loss = ((net(Ftr[b], Etr[b], Otr[b], Ttr[b], Vtr[b], Gtr[b], FGtr[b], Dtr[b],
                         Str[b], Ctr[b]) - Htr[b]) ** 2).mean()
            loss.backward()
            opt.step()

    net.eval()
    res = {"계열": train, "가중": w, "모수": M.n_params(net)}
    with torch.no_grad():
        for tag, idx in (("학습", itr), ("홀드아웃", ite)):
            p = net(T(F[idx]), T(eps[idx]), T(out[idx]), T(tin[idx]), T(val[idx]),
                    T(gate[idx]), T(fgate[idx]), T(duty[idx]),
                    T(dsin[idx]), T(dcos[idx])).cpu().numpy()
            y = hdr[idx]
            r = p - y
            res[f"{tag}_RMSE"] = round(float(np.sqrt((r ** 2).mean())), 3)
            res[f"{tag}_R2"] = round(float(1 - (r ** 2).mean() / y.var()), 4)
            if tag == "홀드아웃":
                res["_resid"] = r
                res["_pred"] = p
                res["β중앙"] = round(float(net.beta(T(val[idx])).median()), 3)
        if w in M.BURIED:
            res["T지중_평균"] = round(float(net.tg[0].item()), 2)
            res["T지중_진폭"] = round(float(np.hypot(net.tg[1].item(), net.tg[2].item())), 2)
        for k, u in enumerate(units):
            res[f"UA배율_{u}"] = round(float(np.exp(net.ua_log[k].item())), 3)
            if w in M.BURIED:
                res[f"k매설_{u}"] = round(float(np.exp(net.log_k[k].item())), 4)
            if w in ("ntu_series", "ntu_flat", "ntu_gated"):
                m = net.flow_from_eps(T(eps[ite]), flat=(w == "ntu_flat"))
                res[f"유량중앙_{u}"] = round(float(m[:, k].median()), 2)
    return res


def residual_structure(r: np.ndarray, d: pd.DataFrame, te: np.ndarray,
                       train: str) -> dict:
    """물리식이 맞으면 잔차에 구조가 남지 않아야 한다."""
    from scipy import stats
    units = config.TRAINS[train]["units"]
    x = d.loc[te, ["t_in", "valve", "doy_sin"] + [f"bath_{u}" for u in units]
                 + [f"eps_{u}" for u in units]]
    s = min(len(r), 200_000)
    sel = np.linspace(0, len(r) - 1, s).astype(int)
    return {f"잔차ρ_{c}": round(float(stats.spearmanr(r[sel], x[c].to_numpy()[sel])[0]), 3)
            for c in x.columns}


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, struct = [], []
    for t in config.TRAINS:
        d = data.build(t)
        tr, te = data.split(d)
        log.info("[%s] 학습 %d행 / 홀드아웃 %d행 (퍼지 %s~%s)",
                 t, tr.sum(), te.sum(), config.TRAIN_END, config.TEST_FROM)
        for w in M.WEIGHTINGS:
            r = _fit(w, t, d, tr, te, dev)
            res = r.pop("_resid")
            r.pop("_pred")
            log.info("  %-12s 홀드아웃 RMSE %.3f℃  R² %.4f  (학습 %.3f)",
                     w, r["홀드아웃_RMSE"], r["홀드아웃_R2"], r["학습_RMSE"])
            if w in ("ntu_series", "ntu_gated", "equal", "hotmax", "valve_only",
                     "duty_weight", "eps_weight", "eps_weight_b", "ntu_series_b"):
                struct.append({"계열": t, "가중": w, **residual_structure(res, d, te, t)})
            rows.append(r)
    out = pd.DataFrame(rows)
    st = pd.DataFrame(struct)
    out.to_csv(config.OUTPUT_DIR / "mixing_test.csv", index=False, encoding="utf-8-sig")
    st.to_csv(config.OUTPUT_DIR / "mixing_residual.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 230)
    print("\n=== 검정 2: 합류점 예측 (홀드아웃 2022~) ===")
    print(out[["계열", "가중", "모수", "학습_RMSE", "홀드아웃_RMSE", "홀드아웃_R2", "β중앙"]]
          .to_string(index=False))
    print("\n=== 학습된 U·A 보정과 역산 유량 ===")
    c = ["계열", "가중"] + [x for x in out.columns if x.startswith(("UA배율", "유량중앙"))]
    print(out[out["가중"].isin(["ntu_series", "ntu_flat", "ntu_gated"])][c].to_string(index=False))
    print("\n=== 잔차 구조 (0 에 가까워야 한다) ===")
    print(st.to_string(index=False))


if __name__ == "__main__":
    main()
