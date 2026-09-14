"""하류 평가 — 정제(AE·KF)가 실제로 도움이 되는가.

정제 도구의 올바른 평가는 **그 출력 자체의 성능**이 아니라
**정제 신호를 넣었을 때 하류 모델이 좋아지는가** 다.

  base        기존 특징만                          ← 현재 M AUC 0.840 / Z 0.792
  +kf         KF 평활값·혁신 추가
  +ae         AE 재구성 오차 추가
  +both       둘 다
  +mode       모드 비중(frac_run/part/idle) 추가    ← 1분 격자에서만 얻어지는 정보
  all         전부

같은 시간분할·같은 보정·같은 지표로 비교한다. 개선이 없으면 "물리 특징이 이미 담고 있었다".
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

HORIZON = 6
KF_COLS = ["kf_t61", "innov_t61", "kf_hdr", "innov_hdr", "kf_tin", "innov_tin",
           "innov_t61_absmax"]
AE_COLS = ["ae_err", "ae_err_max"]
MODE_COLS = ["frac_run", "frac_part", "frac_idle"]


def evaluate(X, ev, tr, va, te, cols) -> dict:
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        random_state=config.RANDOM_SEED, early_stopping=True, validation_fraction=0.15)
    clf.fit(X.loc[tr, cols], ev[tr])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(clf.predict_proba(X.loc[va, cols])[:, 1], np.asarray(ev[va]))
    p = iso.predict(clf.predict_proba(X.loc[te, cols])[:, 1])
    e = np.asarray(ev[te])
    base = float(np.asarray(ev[tr]).mean())
    r = {"AUC": round(float(roc_auc_score(e, p)), 3),
         "Brier": round(float(brier_score_loss(e, p)), 5),
         "Brier개선%": round(100 * (1 - brier_score_loss(e, p)
                                 / brier_score_loss(e, np.full_like(p, base))), 1)}
    for frac in (0.01, 0.05):
        k = max(int(frac * len(p)), 10)
        top = np.argsort(-p)[:k]
        r[f"리프트{int(frac*100)}"] = (round(float(e[top].mean() / e.mean()), 1)
                                     if e.mean() > 0 else np.nan)
    return r


def main() -> None:
    rows = []
    for train in config.TRAINS:
        f = data.build(train)
        ref = pd.read_parquet(config.OUTPUT_DIR / f"refined_{train}.parquet")
        f = f.join(ref, how="left")
        X, y, tr, va, te, cols = data.split(f, HORIZON)
        ev = (y < 0).astype(int)
        base_cols = [c for c in cols if c not in KF_COLS + AE_COLS + MODE_COLS]
        sets = {
            "base": base_cols,
            "+kf": base_cols + [c for c in KF_COLS if c in cols],
            "+ae": base_cols + [c for c in AE_COLS if c in cols],
            "+mode": base_cols + [c for c in MODE_COLS if c in cols],
            "all": cols,
        }
        for name, cs in sets.items():
            cs = [c for c in cs if c in X.columns]
            r = evaluate(X, ev, tr, va, te, cs)
            r.update({"계열": train, "특징집합": name, "특징수": len(cs)})
            rows.append(r)
            log.info("[%s] %-6s (%2d) AUC %.3f Brier개선 %5.1f%% 리프트1%% %4.1f 리프트5%% %4.1f",
                     train, name, len(cs), r["AUC"], r["Brier개선%"], r["리프트1"], r["리프트5"])
    out = pd.DataFrame(rows)[["계열", "특징집합", "특징수", "AUC", "Brier",
                              "Brier개선%", "리프트1", "리프트5"]]
    out.to_csv(config.OUTPUT_DIR / "downstream.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n=== 정제 신호의 하류 기여 (시험 2022-01 이후) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
