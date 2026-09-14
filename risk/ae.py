"""오토인코더 — 정상 운전 재구성 오차를 위험 신호로 쓸 수 있나.

## 두 가지를 본다

  (a) **단독**: 재구성 오차 자체를 위험 점수로 → AUC 가 GBM 에 근접하나
  (b) **결합**: 오차를 GBM 특징으로 추가 → 개선되나

AE 는 라벨을 안 쓴다. 정상 운전을 재구성하도록만 배우고, 못 맞추는 구간을 이상으로 본다.
빙결 위험 구간이 '평소와 다른 운전'이라면 (a)가 작동해야 한다.

## 앞선 AE 파일럿의 버그를 반복하지 않는다 (docs/06 C-6)

  · **StandardScaler 를 쓴다.** robust 스케일러가 `ZI41P_frac`(IQR 0.0037)를 σ 35 로 부풀려
    손실의 99.7% 를 먹은 적이 있다.
  · **스케일러는 훈련구간에서만 적합**한다.
  · 스케일러 통계는 **float64**. float32 누적오차로 평균이 −554.8(실제 −620)이 된 적이 있다.
  · 정상 학습: 훈련구간에서 **사건이 아닌 시점만** 쓴다(사건을 정상으로 배우면 안 된다).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config, data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

HORIZON = 6
EPOCHS = 40
BATCH = 512
LATENT = 6


class AE(nn.Module):
    def __init__(self, d: int, latent: int = LATENT):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 24), nn.ReLU(),
                                 nn.Linear(24, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 24), nn.ReLU(), nn.Linear(24, 64), nn.ReLU(),
                                 nn.Linear(64, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def fit_ae(Xtr: np.ndarray, dev: str, seed: int = config.RANDOM_SEED):
    torch.manual_seed(seed)
    m = AE(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    n = len(t)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = t[perm[i:i + BATCH]]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(b), b)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if ep % 10 == 9:
            log.info("    epoch %2d  loss %.5f", ep + 1, tot / n)
    return m


def recon_error(m, X: np.ndarray, dev: str) -> np.ndarray:
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=dev)
        e = ((m(t) - t) ** 2).mean(dim=1)
    return e.cpu().numpy()


def main() -> None:
    from sklearn.calibration import IsotonicRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", dev)
    rows = []
    for train in config.TRAINS:
        f = data.build(train)
        X, y, tr, va, te, cols = data.split(f, HORIZON)
        ev = (y < 0).astype(int)
        tr_np, va_np, te_np = np.asarray(tr), np.asarray(va), np.asarray(te)

        # 스케일러: 훈련구간, float64 통계
        mu = X[tr].to_numpy(dtype=np.float64).mean(axis=0)
        sd = X[tr].to_numpy(dtype=np.float64).std(axis=0)
        sd[sd < 1e-9] = 1.0
        Z = ((X.to_numpy(dtype=np.float64) - mu) / sd).astype(np.float32)

        normal = tr_np & (np.asarray(ev) == 0)          # 정상만 학습
        log.info("[%s] AE 학습 표본 %d (정상), 특징 %d", train, int(normal.sum()), Z.shape[1])
        m = fit_ae(Z[normal], dev)
        err = recon_error(m, Z, dev)
        ev_te = np.asarray(ev[te])
        base = float(ev[tr].mean())

        # (a) 단독 — 오차를 확률로 쓰려면 보정 필요
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(err[va_np], np.asarray(ev[va]))
        p_solo = iso.predict(err[te_np])

        # (b) 결합 — GBM 특징으로 추가
        X2 = X.copy()
        X2["ae_err"] = err
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
            random_state=config.RANDOM_SEED, early_stopping=True, validation_fraction=0.15)
        clf.fit(X2[tr], ev[tr])
        iso2 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso2.fit(clf.predict_proba(X2[va])[:, 1], np.asarray(ev[va]))
        p_comb = iso2.predict(clf.predict_proba(X2[te])[:, 1])

        for name, p in [("AE 단독", p_solo), ("GBM+AE", p_comb)]:
            r = {"계열": train, "모델": name,
                 "AUC": round(float(roc_auc_score(ev_te, p)), 3),
                 "Brier": round(float(brier_score_loss(ev_te, p)), 5),
                 "Brier개선%": round(100 * (1 - brier_score_loss(ev_te, p)
                                         / brier_score_loss(ev_te, np.full_like(p, base))), 1)}
            for frac in (0.01, 0.05):
                k = max(int(frac * len(p)), 10)
                top = np.argsort(-p)[:k]
                r[f"리프트{int(frac*100)}"] = (round(float(ev_te[top].mean() / ev_te.mean()), 1)
                                              if ev_te.mean() > 0 else np.nan)
            rows.append(r)
            log.info("  [%s] %-8s AUC %.3f Brier개선 %.1f%% 리프트1%% %.1f",
                     train, name, r["AUC"], r["Brier개선%"], r["리프트1"])
    out = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.OUTPUT_DIR / "ae_metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n=== AE 결과 (시험 2022-01 이후) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
