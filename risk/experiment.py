"""문서 15 — 체제 한정 학습 × 열수지 특징 (2×2).

한계 4 (2016 운전 체제 변경): 구 체제(2011~2016-06)를 학습에서 빼면 나아지는가
한계 6 (히터 개별 기여 분리 불가): 열수지 유량·화력·흡수열(문서 12) 특징이 도움이 되는가

네 조합 모두 **시험구간이 같다**(2022-01~, 지평 6h) → 직접 비교.

⚠ `data.load_raw` 는 호출마다 원본 CSV 175개를 다시 읽는다. 계열당 한 번만 열수지 특징을
  포함해 만들고, 열수지를 끄는 조합은 그 열만 뺀다 (8회 → 2회).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, data, model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

HF_PREFIX = ("m_", "fire_", "qgas_", "m_tot", "heat_margin")
H = 6
CASES = [
    ("현행 (전기간·기존특징)", False, None),
    ("＋열수지 특징",          True,  None),
    ("체제한정 (2016-07~)",    False, "2016-07-01"),
    ("체제한정 ＋열수지",       True,  "2016-07-01"),
]
KEEP = ["조합", "계열", "시험n", "학습사건", "학습n", "시험_실제사건률", "ROC_AUC",
        "Brier개선%", "리프트1", "리프트5", "상위1%_사건률", "모델_RMSE", "개선%"]


def main() -> None:
    orig_build = data.build
    config.USE_HEATFLOW = True
    cache = {t: orig_build(t) for t in config.TRAINS}
    for t, f in cache.items():
        log.info("[%s] 캐시 %d행, 열수지 특징 %s", t, len(f),
                 [c for c in f.columns if c.startswith(HF_PREFIX)])

    def build(train: str) -> pd.DataFrame:
        f = cache[train]
        if config.USE_HEATFLOW:
            return f
        return f.drop(columns=[c for c in f.columns if c.startswith(HF_PREFIX)])

    data.build = build
    rows = []
    try:
        for name, use_hf, regime in CASES:
            config.USE_HEATFLOW, config.REGIME_FROM = use_hf, regime
            for t in config.TRAINS:
                X, y, tr, va, te, cols = data.split(build(t), H)
                r = model.run_train(t, H)
                if not r:
                    log.warning("[%s] %s — 표본 부족으로 건너뜀", name, t)
                    continue
                r = {k: v for k, v in r.items() if not k.startswith("_")}
                r.update({"조합": name, "학습n": int(tr.sum()),
                          "학습사건": int((y[tr] < 0).sum()), "특징수": len(cols)})
                rows.append(r)
                log.info("[%s] %s AUC %.3f 리프트1 %s 리프트5 %s | 학습사건 %d / %d",
                         name, t, r.get("ROC_AUC", np.nan), r.get("리프트1"), r.get("리프트5"),
                         r["학습사건"], r["학습n"])
    finally:
        data.build = orig_build
        config.USE_HEATFLOW, config.REGIME_FROM = False, None

    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUT_DIR / "experiment_2x2.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 2×2 (시험 2022~, 지평 6h) ===")
    print(df[[c for c in KEEP + ["특징수"] if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
