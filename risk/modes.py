"""운전 모드 분리 — 빙결 사건은 성격이 다른 둘이 섞여 있다.

## 왜 나누나

`risk/` 1차 구축은 모든 시간을 한 덩어리로 학습시켰다. 그런데 확인해 보니
**학습 데이터의 21.3%(M)·32.5%(Z)가 히터가 아예 안 도는 구간**이고,
**빙결 사건의 55%(M)·27.5%(Z)가 그 정지 구간에서 발생**한다.

두 사건은 물리도 대응도 다르다:
  · **정지 중 빙결** — 히터를 안 돌려서 생긴다. 대응은 "히터를 켜라"
  · **운전 중 빙결** — 돌리는데도 못 따라간다. 대응은 "용량/열화 점검"

하나로 묶어 예측하면 모델이 "히터가 꺼져 있나"만 학습한다. 실제로 GBM 의 중요도에서
`flow_*` 가 상위에 있었고, AE·KF 가 무너진 것도 이 혼합 때문이다
(AE 는 정지를 '정상'으로 배웠고, KF 는 전환점에서 동역학이 바뀌는데 하나의 모형으로 통과했다).

## 모드 정의 (1분 격자, docs/03 관측 레이어 재사용)

    run   계열 안에 `regime == flow` 인 히터가 하나라도 있다      ← 통가스 중
    idle  전부 `isolated`                                        ← 완전 정지
    part  그 사이 (lowflow 만 있거나 일부만 flow)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from prex_multi import config as m_config

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def minute_frame(train: str) -> pd.DataFrame:
    """1분 격자 원본 + 모드 라벨. 정제(AE/KF)는 집계 전 원본에서 해야 한다."""
    spec = config.TRAINS[train]
    units = spec["units"]
    cols = ["Time", "segment_id", spec["hdr"], spec["t_in"], f"beta_{train}", f"valve_{train}"]
    for u in units:
        cols += [f"TI-D2{u}", f"TI33{u}", f"burner_{u}", f"regime_{u}", f"eps_{u}", f"ntu_{u}"]
    if u != "P":
        pass
    g = pd.read_parquet(config.TREND_PARQUET, columns=cols).set_index("Time").sort_index()

    # 출구온도·압력은 prex_multi 산출물에 없다 → 원본에서 읽어 붙인다
    extra_cols = [spec["t61"], spec["p61"], config.P_IN]
    frames = []
    for p in sorted(config.RAW_TREND_DIR.glob("*.csv")):
        d = pd.read_csv(p, usecols=["Time", *extra_cols], low_memory=False)
        d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
        frames.append(d.dropna(subset=["Time"]))
    ex = pd.concat(frames, ignore_index=True).drop_duplicates("Time", keep="last")
    ex = ex.set_index("Time").sort_index().apply(pd.to_numeric, errors="coerce")
    g = g.join(ex, how="left")

    reg = pd.concat([g[f"regime_{u}"] for u in units], axis=1)
    any_flow = (reg == "flow").any(axis=1)
    all_iso = (reg == "isolated").all(axis=1) & reg.notna().all(axis=1)
    mode = pd.Series("part", index=g.index, dtype=object)
    mode[any_flow] = "run"
    mode[all_iso & ~any_flow] = "idle"
    mode[reg.isna().all(axis=1)] = None
    g["mode"] = mode
    return g


def main() -> None:
    pd.set_option("display.width", 210)
    for train in config.TRAINS:
        spec = config.TRAINS[train]
        g = minute_frame(train)
        t61 = g[spec["t61"]]
        ev = (t61 < config.FREEZE_C) & t61.notna()
        ok = g["mode"].notna() & t61.notna()

        print(f"\n=== 계열 {train} ({'+'.join(spec['units'])}) — 1분 격자 {int(ok.sum()):,}행 ===")
        rows = []
        for m in ("run", "part", "idle"):
            k = ok & (g["mode"] == m)
            if k.sum() < 100:
                continue
            e = ev & k
            rows.append({
                "모드": m, "시간비중%": round(100 * float(k.sum() / ok.sum()), 1),
                "행수": int(k.sum()),
                "빙결행": int(e.sum()),
                "모드내 발생률%": round(100 * float(e.sum() / k.sum()), 2),
                "전체사건중 비중%": round(100 * float(e.sum() / (ev & ok).sum()), 1),
                "출구온도 중앙": round(float(t61[k].median()), 2),
                "헤더온도 중앙": round(float(g[spec["hdr"]][k & (g[spec["hdr"]] > 0)].median()), 2),
                "입구온도 중앙": round(float(g[spec["t_in"]][k].median()), 2),
                "압력강하 중앙": round(float((g[config.P_IN] - g[spec["p61"]])[k].median()), 2),
            })
        print(pd.DataFrame(rows).to_string(index=False))

        # 사건 에피소드도 모드별로
        blk = (ev != ev.shift()).cumsum()
        eps = []
        for _, s in g[ev & ok].groupby(blk[ev & ok]):
            eps.append({"len": len(s), "mode": s["mode"].mode().iat[0] if len(s) else None})
        E = pd.DataFrame(eps)
        if len(E):
            print("\n  빙결 에피소드 %d개의 모드 구성:" % len(E))
            print("   " + E.groupby("mode")["len"].agg(
                에피소드="size", 길이중앙="median", 총분="sum").to_string().replace("\n", "\n   "))


if __name__ == "__main__":
    main()
