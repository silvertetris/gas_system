"""제거 실험 — 계절을 빼면 무엇이 남나.

순열 중요도에서 계절·시각이 M 49.8% · Z 39.7% 로 1위였다(`explain.py`).
"겨울이면 위험하다"는 이미 아는 사실이므로, **히터 상태만으로 얼마나 예측되는지**를
따로 재야 실무 가치를 알 수 있다.

특징 집합을 넷으로 나눠 같은 틀로 비교한다:
  full      전부
  no_season 계절·시각 제외          ← 히터·공정 정보만
  season    계절·시각만              ← 날씨 예보 수준의 기준선
  persist   출구온도 이력만          ← 지속성 수준의 기준선
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config, data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SEASON = ["month", "hour", "doy_sin", "doy_cos"]
PERSIST = ["t61", "t61_mean3", "t61_mean12", "t61_mean24",
           "t61_min3", "t61_min12", "t61_min24", "t61_trend6"]


def evaluate(X, ev, tr, va, te, cols) -> dict:
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        random_state=config.RANDOM_SEED, early_stopping=True, validation_fraction=0.15)
    clf.fit(X.loc[tr, cols], ev[tr])
    raw_va, raw_te = (clf.predict_proba(X.loc[va, cols])[:, 1],
                      clf.predict_proba(X.loc[te, cols])[:, 1])
    ev_va, ev_te = ev[va].to_numpy(), ev[te].to_numpy()
    if ev_va.sum() >= 20:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_va, ev_va)
        p = iso.predict(raw_te)
    else:
        p = raw_te
    base = float(ev[tr].mean())
    r = {"AUC": round(float(roc_auc_score(ev_te, p)), 3),
         "Brier": round(float(brier_score_loss(ev_te, p)), 5),
         "Brier개선%": round(100 * (1 - brier_score_loss(ev_te, p)
                                 / brier_score_loss(ev_te, np.full_like(p, base))), 1)}
    for frac in (0.01, 0.05):
        k = max(int(frac * len(p)), 10)
        top = np.argsort(-p)[:k]
        r[f"리프트{int(frac*100)}"] = (round(float(ev_te[top].mean() / ev_te.mean()), 1)
                                     if ev_te.mean() > 0 else np.nan)
    return r


def main() -> None:
    rows = []
    for train in config.TRAINS:
        for h in config.HORIZONS_H:
            f = data.build(train)
            X, y, tr, va, te, cols = data.split(f, h)
            ev = (y < 0).astype(int)
            if ev[te].sum() < 20:
                continue
            sets = {
                "full": cols,
                "no_season": [c for c in cols if c not in SEASON],
                "season": SEASON,
                "persist": PERSIST,
            }
            for name, cs in sets.items():
                cs = [c for c in cs if c in cols]
                r = evaluate(X, ev, tr, va, te, cs)
                r.update({"계열": train, "지평h": h, "특징집합": name, "특징수": len(cs)})
                rows.append(r)
                log.info("[%s h=%d] %-10s (%2d개) AUC %.3f Brier개선 %5.1f%% 리프트1%% %s",
                         train, h, name, len(cs), r["AUC"], r["Brier개선%"], r["리프트1"])
    out = pd.DataFrame(rows)[["계열", "지평h", "특징집합", "특징수", "AUC",
                              "Brier", "Brier개선%", "리프트1", "리프트5"]]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.OUTPUT_DIR / "ablation.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n=== 제거 실험 (시험구간 2022-01 이후) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
