"""운영 가이드 — 경보 임계를 어디에 둘 것인가.

모델을 쓰려면 "확률 몇 % 이상에서 경보를 띄울 것인가"를 정해야 한다.
임계마다 **경보 빈도 / 적중률 / 놓친 사건**이 달라지므로 표로 만들어 현장이 고르게 한다.

  경보율   전체 시간 중 경보가 뜨는 비율 (운영 부담)
  정밀도   경보가 떴을 때 실제로 6시간 내 빙결이 온 비율
  재현율   실제 사건 중 경보로 잡힌 비율
  리프트   정밀도 ÷ 전체 사건률
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config, data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

HORIZON = 6                     # 24h 는 계절 기준선을 못 이긴다(ablation.py) → 6h 만 운영
THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def fit_predict(train: str):
    f = data.build(train)
    X, y, tr, va, te, cols = data.split(f, HORIZON)
    ev = (y < 0).astype(int)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        random_state=config.RANDOM_SEED, early_stopping=True, validation_fraction=0.15)
    clf.fit(X[tr], ev[tr])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(clf.predict_proba(X[va])[:, 1], ev[va].to_numpy())
    p = iso.predict(clf.predict_proba(X[te])[:, 1])
    return p, ev[te].to_numpy(), X.index[te]


def table(p: np.ndarray, y: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    base = float(y.mean())
    days = (idx.max() - idx.min()).days or 1
    rows = []
    for th in THRESHOLDS:
        a = p >= th
        if a.sum() < 5:
            continue
        prec = float(y[a].mean())
        rows.append({
            "임계": th, "경보율%": round(100 * float(a.mean()), 2),
            "경보횟수": int(a.sum()),
            "연간경보시간": round(a.sum() / days * 365, 0),
            "정밀도%": round(100 * prec, 1),
            "재현율%": round(100 * float(y[a].sum() / max(y.sum(), 1)), 1),
            "리프트": round(prec / base, 1) if base > 0 else np.nan,
            "놓친사건": int(y.sum() - y[a].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    alls = []
    for train in config.TRAINS:
        p, y, idx = fit_predict(train)
        t = table(p, y, idx)
        t.insert(0, "계열", train)
        alls.append(t)
        print(f"\n=== 계열 {train} · 6시간 빙결 위험 경보 "
              f"(시험 {idx.min().date()}~{idx.max().date()}, 사건 {int(y.sum())}건 / {len(y)}시간, "
              f"기저율 {100*y.mean():.2f}%) ===")
        print(t.to_string(index=False))
    pd.concat(alls).to_csv(config.OUTPUT_DIR / "operating_guide.csv",
                           index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
