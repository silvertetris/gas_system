"""특징 중요도 — 모델이 물리를 보는가, 계절만 보는가.

순열 중요도(permutation importance)를 **시험구간**에서 AUC 기준으로 잰다.
훈련 중요도(트리 분기 횟수 등)는 과적합된 신호도 크게 나오므로 쓰지 않는다.

판정 기준:
  · 물리 변수(`dp`, `hdr`, `jt_drop`, 히터 상태)가 상위여야 한다
  · 계절 변수(`month`, `doy_*`)만 상위면 **날씨 예보를 다시 만든 것**이지
    히터 위험도 모델이 아니다 — 그 경우 실무 가치가 크게 떨어진다
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from . import config, data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

GROUPS = {
    "압력·JT": ["dp", "p_in", "p61", "jt_drop"],
    "출구온도 이력": ["t61", "t61_mean3", "t61_mean12", "t61_mean24",
                  "t61_min3", "t61_min12", "t61_min24", "t61_trend6"],
    "헤더·입구": ["hdr", "t_in", "t_in_mean24"],
    "히터 상태": [],          # 아래에서 계열별로 채운다
    "계절·시각": ["month", "hour", "doy_sin", "doy_cos"],
}


def run(train: str, h: int, n_repeats: int = 5) -> pd.DataFrame:
    f = data.build(train)
    X, y, tr, va, te, cols = data.split(f, h)
    ev = (y < 0).astype(int)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        random_state=config.RANDOM_SEED, early_stopping=True, validation_fraction=0.15)
    clf.fit(X[tr], ev[tr])
    base = roc_auc_score(ev[te], clf.predict_proba(X[te])[:, 1])
    r = permutation_importance(clf, X[te], ev[te], scoring="roc_auc",
                               n_repeats=n_repeats, random_state=config.RANDOM_SEED, n_jobs=-1)
    out = pd.DataFrame({"특징": cols, "AUC감소": r.importances_mean,
                        "표준편차": r.importances_std}).sort_values("AUC감소", ascending=False)
    out.attrs["base_auc"] = base
    return out


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    for train in config.TRAINS:
        spec = config.TRAINS[train]
        groups = dict(GROUPS)
        groups["히터 상태"] = ([f"bath_{u}" for u in spec["units"]]
                           + [f"out_{u}" for u in spec["units"]]
                           + [f"duty_{u}" for u in spec["units"]]
                           + [f"flow_{u}" for u in spec["units"]] + ["beta", "valve"])
        imp = run(train, 6)
        imp.to_csv(config.OUTPUT_DIR / f"importance_{train}_6h.csv",
                   index=False, encoding="utf-8-sig")
        print(f"\n=== 계열 {train} · 지평 6h (기준 AUC {imp.attrs['base_auc']:.3f}) ===")
        print("\n[상위 12개 특징]")
        print(imp.head(12).round(4).to_string(index=False))
        print("\n[그룹별 합계]")
        rows = []
        for gname, members in groups.items():
            k = imp[imp["특징"].isin(members)]
            rows.append({"그룹": gname, "특징수": len(k),
                         "AUC감소합": round(float(k["AUC감소"].sum()), 4)})
        gdf = pd.DataFrame(rows).sort_values("AUC감소합", ascending=False)
        tot = gdf["AUC감소합"].sum()
        gdf["비중%"] = (100 * gdf["AUC감소합"] / tot).round(1) if tot > 0 else np.nan
        print(gdf.to_string(index=False))


if __name__ == "__main__":
    main()
