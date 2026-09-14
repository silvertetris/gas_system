"""빙결 여유 예측 + 위험확률. 기준선 3종을 반드시 같이 낸다.

⚠ **기준선 없이 성능을 보고하면 안 된다.** 앞선 LSTM 파일럿이 persistence 대비 RMSE 를
20.4% 개선했지만 정작 고장 탐지 ROC-AUC 는 0.419(무작위 이하)였다. 출구온도는 자기상관이
매우 강해서 **아무 모델이나 그럴듯해 보인다.**

기준선:
  · unconditional — 훈련구간 여유의 중앙값을 항상 예측
  · persistence   — 현재 출구온도를 그대로 예측
  · seasonal      — 같은 월·시각의 훈련구간 중앙값

위험확률은 **분류기로 직접** 추정하고 **검증구간에서 등장성 보정(isotonic)** 한다.

⚠ 1차 시도에서는 훈련 잔차 경험분포로 확률을 냈는데 **캘리브레이션이 크게 어긋났다**
  (예측 평균 3.2% vs 실제 0.91%, 3.5배 과대). 사건이 2016년 이후 급감해서
  훈련구간(사건 많음)의 잔차분포가 시험구간(사건 드묾)에 맞지 않기 때문이다.
  → 회귀 잔차가 아니라 **사건 자체를 분류**하고, 훈련에 쓰지 않은 **검증구간으로 보정**한다.
  순위(AUC·리프트)는 1차에서도 좋았으므로 문제는 확률의 눈금이었다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config, data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = y - p
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e)))}


def calibration_table(p: np.ndarray, y: np.ndarray, bins=(0, .01, .05, .1, .2, .5, 1.01)) -> str:
    """예측확률 구간별 실제 발생률. 확률로 쓰려면 이 둘이 맞아야 한다."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (p >= lo) & (p < hi)
        if k.sum() < 20:
            continue
        out.append(f"{lo:.2f}~{hi:.2f}: n={int(k.sum()):6d} 예측 {p[k].mean():.4f} 실제 {y[k].mean():.4f}")
    return " | ".join(out)


def run_train(train: str, h: int) -> dict:
    f = data.build(train)
    X, y, tr, va, te, cols = data.split(f, h)
    log.info("[%s h=%d] 표본 %d | 훈련 %d 검증 %d 시험 %d | 특징 %d",
             train, h, len(X), tr.sum(), va.sum(), te.sum(), len(cols))
    if te.sum() < 500 or tr.sum() < 5000:
        return {}

    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=config.RANDOM_SEED,
        early_stopping=True, validation_fraction=0.15)
    m.fit(X[tr], y[tr])

    out = {"계열": train, "지평h": h, "시험n": int(te.sum())}
    pred_te = m.predict(X[te])
    out.update({f"모델_{k}": round(v, 3) for k, v in _metrics(y[te].to_numpy(), pred_te).items()})

    # 기준선
    base_u = np.full(te.sum(), float(y[tr].median()))
    base_p = X.loc[te, "t61"].to_numpy() - config.FREEZE_C
    key = pd.MultiIndex.from_arrays([X.index[tr].month, X.index[tr].hour])
    seas = pd.Series(y[tr].to_numpy(), index=key).groupby(level=[0, 1]).median()
    base_s = pd.MultiIndex.from_arrays([X.index[te].month, X.index[te].hour]).map(seas).to_numpy()
    base_s = np.where(np.isfinite(base_s.astype(float)), base_s, float(y[tr].median())).astype(float)
    for name, b in [("무조건부", base_u), ("지속성", base_p), ("계절", base_s)]:
        out.update({f"{name}_{k}": round(v, 3) for k, v in _metrics(y[te].to_numpy(), b).items()})

    best_base = min(out["무조건부_RMSE"], out["지속성_RMSE"], out["계절_RMSE"])
    out["개선%"] = round(100 * (1 - out["모델_RMSE"] / best_base), 1)

    # --- 위험확률: 분류기 + 검증구간 등장성 보정
    ev_all = (y < 0).astype(int)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=config.RANDOM_SEED,
        early_stopping=True, validation_fraction=0.15)
    clf.fit(X[tr], ev_all[tr])
    raw_va, raw_te = clf.predict_proba(X[va])[:, 1], clf.predict_proba(X[te])[:, 1]
    ev_va, ev_te = ev_all[va].to_numpy(), ev_all[te].to_numpy()

    if ev_va.sum() >= 20:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_va, ev_va)
        p_te = iso.predict(raw_te)
    else:                                   # 검증구간에 사건이 없으면 보정 불가
        p_te = raw_te
        log.warning("[%s h=%d] 검증구간 사건 %d건 — 보정 생략", train, h, int(ev_va.sum()))

    out["시험_실제사건률"] = round(float(ev_te.mean()), 5)
    out["보정전_평균확률"] = round(float(raw_te.mean()), 5)
    out["보정후_평균확률"] = round(float(p_te.mean()), 5)
    if ev_te.sum() >= 5:
        out["ROC_AUC"] = round(float(roc_auc_score(ev_te, p_te)), 3)
        out["Brier"] = round(float(brier_score_loss(ev_te, p_te)), 5)
        # 기준선 Brier: 항상 훈련 사건률을 예측
        base_p = float(ev_all[tr].mean())
        out["Brier_기준선"] = round(float(brier_score_loss(ev_te, np.full_like(p_te, base_p))), 5)
        out["Brier개선%"] = round(100 * (1 - out["Brier"] / out["Brier_기준선"]), 1)
    for frac in (0.01, 0.05):
        k = max(int(frac * len(p_te)), 10)
        top = np.argsort(-p_te)[:k]
        out[f"상위{int(frac*100)}%_사건률"] = round(float(ev_te[top].mean()), 4)
        out[f"리프트{int(frac*100)}"] = (round(float(ev_te[top].mean() / ev_te.mean()), 1)
                                       if ev_te.mean() > 0 else np.nan)
    out["_calib"] = calibration_table(p_te, ev_te)
    return out


def main() -> None:
    rows = []
    for train in config.TRAINS:
        for h in config.HORIZONS_H:
            r = run_train(train, h)
            if r:
                calib = r.pop("_calib", "")
                rows.append(r)
                log.info("  → RMSE %.3f vs 기준선 %.3f (%.1f%%) | 사건률 실제 %.4f "
                         "보정전 %.4f 보정후 %.4f | AUC %s Brier개선 %s%% | 리프트1%% %s",
                         r["모델_RMSE"], min(r["무조건부_RMSE"], r["지속성_RMSE"], r["계절_RMSE"]),
                         r["개선%"], r["시험_실제사건률"], r["보정전_평균확률"],
                         r["보정후_평균확률"], r.get("ROC_AUC", "-"),
                         r.get("Brier개선%", "-"), r.get("리프트1", "-"))
                log.info("     캘리브레이션: %s", calib)
    out = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.OUTPUT_DIR / "risk_metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n=== 빙결 여유 예측 (시험구간 2022-01 이후) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
