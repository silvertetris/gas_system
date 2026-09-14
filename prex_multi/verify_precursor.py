"""알람 선행/사후 징후 검정 — AE 로 넘어가기 전 선결 확인.

## 왜 먼저 하나

열화지표에서 배운 것: **지표를 만들고 나서 대조군을 돌리면 늦다.** `U·A` 는 정비 4건 전부
회복하는 것처럼 보였지만 아무 시점이나 잡아도 같은 크기로 움직였다(docs/07).
AE 도 재구성 오차는 반드시 나온다 — 문제는 그게 **알람과 관계있는 오차인가** 다.

## 무엇을 보나

알람 상승엣지 기준으로 구간을 나눠, 각 구간의 관측량이 **무작위 시점의 같은 길이 구간**과
구별되는지 본다.

    [-24h,-6h]  [-6h,-1h]  [-1h,0]  ← 선행 (예측 가능성)
    [0,+1h]                          ← 사후 (탐지 가능성)

선행 구간이 구별되면 AE 가 **예측**할 여지가 있다. 사후만 구별되면 **탐지**에 그친다
(사용자 확인: 사후 탐지도 가치 있음). 둘 다 아니면 그 알람은 이 계측으로 다룰 수 없다.

판정은 `verify_signal.py` 와 같은 틀 — **단측 순위검정**. 정규성 가정 없이,
알람 구간의 지표값이 대조군 분포의 어느 분위에 있는지로 본다.

⚠ `TAHD1x`(수조 수온 High)는 제외한다 — 수조 5.9℃ 에서 발화한 기록이 있어 신뢰할 수 없다(docs/05).
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

LEAD_WINDOWS = [("-24h~-6h", -24, -6), ("-6h~-1h", -6, -1),
                ("-1h~0", -1, 0), ("0~+1h", 0, 1)]
N_NULL = 400
NULL_GUARD_H = 24 * 7          # 알람에서 이만큼 떨어진 시점만 대조군
MIN_ROWS = 30                  # 구간당 최소 1분 표본
MIN_EVENTS = 60
UNRELIABLE = {"bath_h"}        # TAHD1x — docs/05
TOP_N_ALARMS = 3


def alarm_edges(tag: str) -> pd.Series:
    """DI 3함정(중복·시각없음·비트팩) 보정 후 상승엣지 시각. docs/02 2절."""
    for f in sorted(glob.glob(str(config.ALARM_DI_DIR / "*.csv"))):
        if tag not in pd.read_csv(f, nrows=0).columns:
            continue
        d = pd.read_csv(f, usecols=["Time", tag], dtype=str)
        x = d[["Time", tag]].dropna()
        x = x[~x["Time"].str.contains("n/a")]
        v = pd.to_numeric(x[tag], errors="coerce").astype("Int64") & 1
        x = x.assign(v=v).dropna(subset=["v"])
        x["Time"] = pd.to_datetime(x["Time"])
        x = x.drop_duplicates(["Time", "v"]).sort_values("Time")
        return x[x["v"].diff() != 0].pipe(lambda s: s[s["v"] == 1])["Time"].reset_index(drop=True)
    return pd.Series(dtype="datetime64[ns]")


def feature_frame(g: pd.DataFrame, u: str) -> pd.DataFrame:
    tr = config.TRAIN[u]
    f = pd.DataFrame({
        "eps": g[f"eps_{u}"], "ntu": g[f"ntu_{u}"], "drive": g[f"drive_{u}"],
        "dT": g[f"dT_{u}"], "beta": g[f"beta_{tr}"], "duty": g[f"burner_{u}"],
        "t_bath": g[f"TI-D2{u}"], "t_in": g[config.TRAIN_INLET[tr]],
        "flow": (g[f"regime_{u}"] == "flow").astype(float),
        "iso": (g[f"regime_{u}"] == "isolated").astype(float),
    })
    if u != "P":
        f["level"] = g[f"LI-D1{u}"]
    return f


def _agg(F: pd.DataFrame, t: pd.Timestamp, h0: int, h1: int) -> pd.Series | None:
    s = F.loc[t + pd.Timedelta(hours=h0): t + pd.Timedelta(hours=h1)]
    if len(s) < MIN_ROWS:
        return None
    return s.mean(numeric_only=True)


def run() -> pd.DataFrame:
    g = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet").set_index("Time").sort_index()
    rng = np.random.default_rng(0)
    rows = []
    for u in config.UNITS:
        F = feature_frame(g, u)
        tags = {k: v for k, v in config.alarm_tags(u).items()
                if k not in ("running",) and k not in UNRELIABLE}
        counts = {k: len(alarm_edges(v)) for k, v in tags.items()}
        top = sorted(counts, key=counts.get, reverse=True)[:TOP_N_ALARMS]
        for key in top:
            ev = alarm_edges(tags[key])
            ev = ev[(ev >= F.index.min() + pd.Timedelta(hours=48))
                    & (ev <= F.index.max() - pd.Timedelta(hours=2))]
            if len(ev) < MIN_EVENTS:
                continue
            # 대조군: 알람에서 7일 이상 떨어진 시점
            ev_np = ev.to_numpy()
            nulls = []
            lo = F.index.min() + pd.Timedelta(hours=48)
            span = (F.index.max() - pd.Timedelta(hours=2) - lo).total_seconds()
            while len(nulls) < N_NULL:
                t = lo + pd.Timedelta(seconds=float(rng.random() * span))
                if np.min(np.abs((ev_np - np.datetime64(t)).astype("timedelta64[h]").astype(float))) >= NULL_GUARD_H:
                    nulls.append(t)
            for lab, h0, h1 in LEAD_WINDOWS:
                A = pd.DataFrame([r for r in (_agg(F, t, h0, h1) for t in ev) if r is not None])
                N = pd.DataFrame([r for r in (_agg(F, t, h0, h1) for t in nulls) if r is not None])
                if len(A) < MIN_EVENTS // 2 or len(N) < 50:
                    continue
                for col in A.columns:
                    a, n = A[col].dropna(), N[col].dropna()
                    if len(a) < 20 or len(n) < 30 or n.std() == 0:
                        continue
                    med_a, med_n = float(a.median()), float(n.median())
                    # 단측 순위: 알람 구간 중앙값이 대조군 분포의 어느 꼬리에 있나
                    p = float(min(np.mean(n <= med_a), np.mean(n >= med_a))) * 2
                    rows.append({"히터": u, "알람": key, "구간": lab, "지표": col,
                                 "n_알람": len(a), "알람중앙": round(med_a, 3),
                                 "대조중앙": round(med_n, 3),
                                 "효과크기": round((med_a - med_n) / (n.std() + 1e-12), 2),
                                 "p": round(p, 4)})
        log.info("[%s] 상위 알람 %s 검정 완료", u, top)
    return pd.DataFrame(rows)


def main() -> None:
    out = run()
    out.to_csv(config.OUTPUT_DIR / "verify_precursor.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    sig = out[(out["p"] < 0.01) & (out["효과크기"].abs() > 0.5)]
    print(f"\n=== 검정 {len(out)}개 중 p<0.01 & |효과|>0.5 인 것 {len(sig)}개 ===")
    print("\n[선행 구간]")
    lead = sig[sig["구간"] != "0~+1h"].sort_values("p")
    print(lead.head(25).to_string(index=False) if len(lead) else "  없음")
    print("\n[사후 구간 0~+1h]")
    post = sig[sig["구간"] == "0~+1h"].sort_values("p")
    print(post.head(20).to_string(index=False) if len(post) else "  없음")
    print("\n=== 구간별 유의 건수 ===")
    print(out.assign(sig=(out["p"] < 0.01) & (out["효과크기"].abs() > 0.5))
             .groupby("구간")["sig"].agg(검정수="size", 유의="sum").to_string())


if __name__ == "__main__":
    main()
