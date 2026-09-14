"""4대 히터 전처리 진입점.

    .venv/bin/python -m prex_multi.pipeline

출력 (prex_multi/output/, git 미포함):
    trend_all.parquet    4대 공통 1분 그리드 + 계열 파생(beta/헤더온도) + 히터별 파생(dT/drive/eps/ntu/regime)
    summary.csv          히터별 데이터 요약 (운전상태 분포 포함)
    train_summary.csv    계열별 바이패스 요약
"""
from __future__ import annotations

import logging

import pandas as pd

from . import config, trend

logger = logging.getLogger("prex_multi")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = trend.run()
    out = config.OUTPUT_DIR / "trend_all.parquet"
    grid.reset_index().to_parquet(out, index=False)
    logger.info("저장: %s (%d행 × %d열)", out, len(grid), grid.shape[1])

    rows = []
    for u in config.UNITS:
        t = config.trend_tags(u)
        dT = grid[f"dT_{u}"]
        rows.append({
            "히터": f"HTR-31{u}", "방식": config.BATH_TYPE[u], "제작사": config.MAKER[u],
            "설계유량_Nm3h": config.DESIGN_FLOW_NM3_H[u],
            "설계m_gas_kgs": round(config.design_mgas_kg_s(u), 2),
            "설계열량_MW": round(config.design_duty_w(u) / 1e6, 3),
            "UA_추정_kWK": round(config.scaled_ua_w_per_k(u) / 1e3, 1),
            "수위계": "있음" if t["level"] else "없음",
            "T_in태그": t["T_in"],
            "가동률_%": round(100 * (dT > 5).sum() / max(dT.notna().sum(), 1), 1),
            "일한일수": round(int((dT > 5).sum()) / 1440),
            "eps중앙": round(float(grid[f"eps_{u}"].median()), 3),
            "eps유효_%": round(100 * grid[f"eps_{u}"].notna().sum() / len(grid), 1),
        })
        # 운전상태 분포 — 열전달 해석이 유효한 구간이 얼마나 되는지가 핵심 지표다(§15).
        vc = grid[f"regime_{u}"].value_counts()
        n = max(int(vc.sum()), 1)
        for r in trend.REGIMES:
            rows[-1][f"{r}_%"] = round(100 * int(vc.get(r, 0)) / n, 1)
        rows[-1]["통가스_일수"] = round(int(vc.get("flow", 0)) / 1440)
    summary = pd.DataFrame(rows)
    summary.to_csv(config.OUTPUT_DIR / "summary.csv", index=False)

    trows = []
    for tr, units in config.TRAIN_UNITS.items():
        b = grid[f"beta_{tr}"]
        v = grid[f"valve_{tr}"]
        k = b.notna() & v.notna()
        trows.append({
            "계열": tr, "히터": "+".join(units),
            "입구": config.TRAIN_INLET[tr], "헤더": config.TRAIN_HEADER[tr],
            "밸브": config.TRAIN_BYPASS_VALVE[tr],
            "β유효_%": round(100 * b.notna().sum() / len(grid), 1),
            "β중앙": round(float(b.median()), 3),
            "β_p10": round(float(b.quantile(0.10)), 3),
            "β_p90": round(float(b.quantile(0.90)), 3),
            "밸브개도_중앙": round(float(v.median()), 1),
            "β_vs_밸브_Spearman": round(float(b[k].corr(v[k], method="spearman")), 3),
        })
    tsummary = pd.DataFrame(trows)
    tsummary.to_csv(config.OUTPUT_DIR / "train_summary.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== 계열(바이패스) ===")
    print(tsummary.to_string(index=False))
    print("\n=== 히터 ===")
    print(summary.to_string(index=False))
    logger.info("=== 완료 ===")


if __name__ == "__main__":
    main()
