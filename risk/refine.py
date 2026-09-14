"""정제(refinement) 단계 — AE·KF 의 **제자리**.

## 역할을 바로잡는다

AE·KF 는 **예측기가 아니라 PINN/하류 모델 이전의 정제 도구**다. 1차 시도에서 이 둘을
위험확률 예측기로 세워 GBM 과 경쟁시킨 것은 설계 착오였다. 여기서는 원래 역할로 되돌린다:

  · **KF** — 관측 잡음 제거. 평활된 수준(level)과 **혁신(innovation = 관측 − 사전예측)** 을 낸다.
    혁신이 크다 = 그 시점 관측이 최근 동역학에서 튄다 = **계기 이상이거나 급변**.
  · **AE** — 다변수 조합의 정상성. 재구성 오차가 크다 = **평소 안 나오는 센서값 조합**.

출력은 예측값이 아니라 **하류 모델의 입력**이다. 따라서 평가도 "AE 의 AUC" 가 아니라
**"정제 신호를 넣으면 하류가 좋아지는가"** 로 한다.

## 모드별로 한다 (이것이 1차 실패의 직접 원인이었다)

1분 격자 실측(docs `risk/modes.py`):
  · 계열 M — run 72.1% / part 27.2% / idle 0.8%,  **모드내 빙결 발생률 1.89 / 10.46 / 3.39%**
  · 계열 Z — run 54.6% / part 40.0% / idle 5.4%,  **1.65 / 1.71 / 2.68%**

`part`(저유량·일부 통가스)의 발생률이 M 에서 `run` 의 **5.5배**다. 모드를 섞어 학습하면
AE 는 "어느 모드인가"를 재구성 오차로 뱉고 **모드 안의 이탈을 못 본다.**
→ AE 는 **모드별로 따로 적합**하고, KF 는 **segment + 모드 전환에서 리셋**한다.

⚠ **1분 원본에서 한다.** 1시간 집계는 버너 사이클(ON 8~18분 / OFF 41~59분)을 뭉개고,
   실제로 1시간 집계로 모드를 판정했을 때 `idle` 비중이 0.8% → 21.3% 로 왜곡됐다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from scipy.signal import lfilter
from torch import nn

from . import config, modes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

KF_GAIN = 0.05             # 정상상태 칼만이득 (국소수준 모형). 1분 격자에서 시정수 ≈ 20분
AE_EPOCHS = 25
AE_BATCH = 4096
AE_LATENT = 5
MODES = ("run", "part")    # idle 은 가스가 거의 안 흐른다 — 제외


def kf_innovation(y: np.ndarray, block: np.ndarray, gain: float = KF_GAIN):
    """정상상태 국소수준 KF. 블록(세그먼트+모드) 경계에서 리셋.

    x_t = x_{t-1} + K·(y_t − x_{t-1})  →  EWMA. `lfilter` 로 벡터화한다
    (6백만 점을 파이썬 루프로 돌 수 없다). 반환: 평활값, 혁신(= y − 사전예측).
    """
    lvl = np.full(len(y), np.nan)
    innov = np.full(len(y), np.nan)
    valid = np.isfinite(y)
    # 블록 시작 인덱스
    starts = np.flatnonzero(np.r_[True, block[1:] != block[:-1]])
    ends = np.r_[starts[1:], len(y)]
    for s, e in zip(starts, ends):
        seg = y[s:e]
        m = np.isfinite(seg)
        if m.sum() < 5:
            continue
        z = np.where(m, seg, np.nan)
        z = pd.Series(z).ffill().bfill().to_numpy()          # 필터 내부에서만 채운다
        sm = lfilter([gain], [1.0, -(1.0 - gain)], z, zi=[z[0] * (1.0 - gain)])[0]
        prior = np.r_[z[0], sm[:-1]]                          # 사전예측 = 직전 평활값
        lvl[s:e] = np.where(m, sm, np.nan)
        innov[s:e] = np.where(m, z - prior, np.nan)
    return lvl, innov


class AE(nn.Module):
    def __init__(self, d: int, latent: int = AE_LATENT):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def fit_ae(Z: np.ndarray, dev: str, tag: str) -> AE:
    torch.manual_seed(config.RANDOM_SEED)
    m = AE(Z.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    t = torch.tensor(Z, dtype=torch.float32, device=dev)
    for ep in range(AE_EPOCHS):
        perm = torch.randperm(len(t), device=dev)
        tot = 0.0
        for i in range(0, len(t), AE_BATCH):
            b = t[perm[i:i + AE_BATCH]]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(b), b)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if ep % 12 == 11:
            log.info("      [%s] epoch %2d loss %.5f", tag, ep + 1, tot / len(t))
    return m


def _kf_part(g: pd.DataFrame, spec: dict, train: str) -> pd.DataFrame:
    """KF: 핵심 신호 3종, 세그먼트+모드 블록 단위. 고정 이득이라 **적합할 모수가 없다** →
    fold 경계가 달라도 한 번만 계산하면 된다."""
    blk = (g["segment_id"].astype(str) + "|" + g["mode"].astype(str)).to_numpy()
    blk_id = pd.factorize(blk)[0]
    out = pd.DataFrame(index=g.index)
    for name, col in [("t61", spec["t61"]), ("hdr", spec["hdr"]), ("tin", spec["t_in"])]:
        y = g[col].to_numpy(dtype=float)
        if name == "hdr":
            y = np.where(y > 0, y, np.nan)                     # 2013-06 이전 상수 0
        lvl, inn = kf_innovation(y, blk_id)
        out[f"kf_{name}"] = lvl
        out[f"innov_{name}"] = inn
        log.info("[%s] KF %-4s 평활 %d행, 혁신 |중앙| %.4f",
                 train, name, int(np.isfinite(lvl).sum()), float(np.nanmedian(np.abs(inn))))
    return out


def _ae_part(g: pd.DataFrame, spec: dict, train: str, fit_end, dev: str) -> pd.Series:
    """AE: 모드별로 따로, **`fit_end` 이전 1분 데이터로만** 적합하고 전 구간에 적용한다."""
    # ⚠ `eps_u` 는 넣지 않는다 — 히터가 `isolated` 면 정의되지 않아 NaN 이고,
    #   한 대만 차단돼도 행 전체가 빠져 커버리지가 무너진다(1차 시도: M 47% · Z 22%).
    #   모드별 AE 의 목적은 **센서값 조합의 정상성**이므로 원시 계측만으로 충분하다.
    feat = ([f"TI-D2{u}" for u in spec["units"]] + [f"TI33{u}" for u in spec["units"]]
            + [f"burner_{u}" for u in spec["units"]]
            + [spec["hdr"], spec["t_in"], spec["t61"], spec["p61"], config.P_IN,
               f"valve_{train}"])
    F = g[feat].apply(pd.to_numeric, errors="coerce")
    F[spec["hdr"]] = F[spec["hdr"]].where(F[spec["hdr"]] > 0)
    ae_err = pd.Series(np.nan, index=g.index)
    train_end = pd.Timestamp(fit_end)
    for md in MODES:
        k = (g["mode"] == md) & F.notna().all(axis=1)
        fit_k = k & (g.index < train_end)                      # 훈련구간만으로 적합
        if fit_k.sum() < 5000:
            log.info("[%s] AE mode=%s 표본 %d — 생략", train, md, int(fit_k.sum()))
            continue
        A = F[fit_k].to_numpy(dtype=np.float64)
        mu, sd = A.mean(axis=0), A.std(axis=0)
        sd[sd < 1e-9] = 1.0                                     # float64 통계 (docs/06 C-6)
        Z = ((F[k].to_numpy(dtype=np.float64) - mu) / sd).astype(np.float32)
        Zfit = ((A - mu) / sd).astype(np.float32)
        log.info("[%s] AE mode=%-4s 적합 %d행 / 적용 %d행, 특징 %d (적합 끝 %s)",
                 train, md, len(Zfit), len(Z), Z.shape[1], train_end.date())
        m = fit_ae(Zfit, dev, f"{train}/{md}")
        with torch.no_grad():
            t = torch.tensor(Z, dtype=torch.float32, device=dev)
            e = ((m(t) - t) ** 2).mean(dim=1).cpu().numpy()
        ae_err[k] = e
    return ae_err


def _hourly(kf: pd.DataFrame, ae_err: pd.Series, mode: pd.Series, freq: str | None = None) -> pd.DataFrame:
    """집계 (하류 입력 형태). 기본 주기는 risk config.FREQ(1h). 컬럼 순서는 이전 판과 같다."""
    out = kf.copy()
    out["ae_err"] = ae_err
    idx = out.index.floor(freq or config.FREQ)
    H = out.groupby(idx).mean()
    H["ae_err_max"] = ae_err.groupby(idx).max()
    H["innov_t61_absmax"] = kf["innov_t61"].abs().groupby(idx).max()
    for md in ("run", "part", "idle"):
        H[f"frac_{md}"] = (mode == md).groupby(idx).mean()
    return H


def build(train: str, fit_end=None) -> pd.DataFrame:
    """1분 격자에서 정제 신호를 만들고 1시간으로 집계해 반환. `fit_end` 기본은 TRAIN_END."""
    spec = config.TRAINS[train]
    g = modes.minute_frame(train)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fit_end = fit_end or config.TRAIN_END
    return _hourly(_kf_part(g, spec, train), _ae_part(g, spec, train, fit_end, dev), g["mode"])


def build_many(train: str, fit_ends: dict, out_dir, freq: str | None = None) -> dict:
    """fold 경계별 정제 산출 — 1분 원본 적재·KF 는 **한 번만**, AE 만 경계마다 재적합.

    `fit_ends` = {태그: 적합 종료시각}. 산출 `out_dir/refined_{train}_{태그}.parquet`.
    """
    import pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = config.TRAINS[train]
    g = modes.minute_frame(train)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kf = _kf_part(g, spec, train)
    paths = {}
    for tag, fe in fit_ends.items():
        H = _hourly(kf, _ae_part(g, spec, train, fe, dev), g["mode"], freq)
        path = out_dir / f"refined_{train}_{tag}.parquet"
        H.to_parquet(path)
        paths[tag] = path
        log.info("[%s] %s (AE 적합 끝 %s) 저장 %s", train, tag, fe, path.name)
    return paths


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for train in config.TRAINS:
        H = build(train)
        p = config.OUTPUT_DIR / f"refined_{train}.parquet"
        H.to_parquet(p)
        log.info("[%s] 저장 %s  %s", train, p.name, H.shape)
        print(f"\n=== 계열 {train} 정제 신호 요약 ===")
        print(H.describe().T[["count", "mean", "50%", "max"]].round(4).to_string())


if __name__ == "__main__":
    main()
