"""태그별 EDA 진입점:  .venv/bin/python -m eda.pipeline

원본 Trend CSV 175개월의 **모든 컬럼**을 훑어 태그마다 PNG 한 장을 만든다.
태그사전(`data/docs/02_KPOS_태그사전.md`)에서 설비·부위·측정량·역할을 가져와 그림에 싣는다.

산출물:
    eda/output/tags/<태그>.png      태그별 상세 (LIVE 태그)
    eda/output/tags_dead/<태그>.png  0 아닌 값이 희소한 태그 (참고용)
    eda/output/tag_index.csv         전체 목록 + 통계 + 메타
    eda/output/INDEX.md              사람이 읽는 색인
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, plots, tagmeta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ⚠ 전 컬럼을 한 번에 올리면 안 된다. 7.4M행 × 134열 × float64 ≈ 8 GB 이고
#   `apply(to_numeric)` 가 사본을 또 만들어 메모리가 터진다(1차 시도에서 무출력 종료).
#   → **컬럼 그룹 단위로 나눠 읽고, float32 로 받는다.**
GROUP_SIZE = 12


def all_columns() -> list[str]:
    paths = sorted(config.RAW_TREND_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Trend CSV 없음: {config.RAW_TREND_DIR}")
    cols = list(pd.read_csv(paths[0], nrows=0).columns)
    return [c for c in cols if c != "Time"]


def load_group(cols: list[str]) -> pd.DataFrame:
    """지정 컬럼만 전 기간 로드. 1분 격자 reindex 는 하지 않는다(원본 관측 그대로)."""
    paths = sorted(config.RAW_TREND_DIR.glob("*.csv"))
    frames = []
    for p in paths:
        d = pd.read_csv(p, usecols=["Time", *cols], low_memory=False)
        d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
        d = d.dropna(subset=["Time"])
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("Time", keep="last").set_index("Time").sort_index()


def main() -> None:
    live_dir = config.FIGURE_DIR
    dead_dir = config.OUTPUT_DIR / "tags_dead"
    for d in (live_dir, dead_dir):
        d.mkdir(parents=True, exist_ok=True)

    cols = all_columns()
    meta_all = tagmeta.load()
    log.info("컬럼 %d개, 태그사전 %d개 로드", len(cols), len(meta_all))

    rows = []
    i = 0
    for gi in range(0, len(cols), GROUP_SIZE):
        grp = cols[gi:gi + GROUP_SIZE]
        df = load_group(grp)
        log.info("그룹 %d/%d 로드: %d행 × %d열 (%s …)",
                 gi // GROUP_SIZE + 1, (len(cols) - 1) // GROUP_SIZE + 1,
                 len(df), len(grp), grp[0])
        for tag in grp:
            i += 1
            s = df[tag]
            v = s.dropna()
            nz = v[v != 0]
            dead = len(nz) < config.DEAD_NONZERO_MAX
            out_dir = dead_dir if dead else live_dir
            try:
                path = plots.plot_tag(tag, s, out_dir)
            except Exception as e:                  # 한 태그가 실패해도 전체를 멈추지 않는다
                log.warning("[%s] 그림 실패: %s", tag, e)
                path = None
            m = tagmeta.describe(tag)
            rows.append({
                "태그": tag, "상태": "DEAD" if dead else "LIVE",
                "설비": m["설비"], "절": m["절"], "설명": m["설명"], "호기": m["호기"],
                "측정량": m["측정량"], "계기기능": m["계기기능"], "역할": m["역할"],
                "신호": m["신호"], "배선": m["배선"], "사전상태": m["상태"],
                "주의": m["주의"],
                "관측행": len(v), "0아닌값": len(nz),
                "결측률%": round(100 * (1 - len(v) / max(len(s), 1)), 2),
                "중앙": round(float(v.median()), 4) if len(v) else np.nan,
                "최소": round(float(v.min()), 3) if len(v) else np.nan,
                "최대": round(float(v.max()), 3) if len(v) else np.nan,
                "고유값": int(v.nunique()),
                "파일": path.name if path else "",
            })
        del df
        log.info("  누적 %3d/%d 태그 처리", i, len(cols))

    idx = pd.DataFrame(rows)
    idx.to_csv(config.OUTPUT_DIR / "tag_index.csv", index=False, encoding="utf-8-sig")

    # 사람이 읽는 색인
    live = idx[idx["상태"] == "LIVE"].sort_values(["설비", "절", "태그"])
    dead = idx[idx["상태"] == "DEAD"].sort_values("태그")
    lines = ["# 태그별 EDA 색인", "",
             f"> 생성 자동. 원본 Trend 175개월 전 컬럼 {len(idx)}개 중 "
             f"**LIVE {len(live)}개 · DEAD {len(dead)}개**.",
             "> 태그 메타는 `data/docs/02_KPOS_태그사전.md`(IO LIST 기반)에서 가져왔다.", "",
             "## LIVE — 실측값이 있는 태그", ""]
    cur = None
    for _, r in live.iterrows():
        key = (r["설비"], r["절"])
        if key != cur:
            lines += ["", f"### {r['설비']} › {r['절'] or '(구분 없음)'}", "",
                      "| 태그 | 설명 | 측정량 | 중앙 | 범위 | 결측% | 그림 |",
                      "|---|---|---|---|---|---|---|"]
            cur = key
        lines.append(f"| `{r['태그']}` | {r['설명']} | {r['측정량']} | {r['중앙']} | "
                     f"{r['최소']} ~ {r['최대']} | {r['결측률%']} | "
                     f"[png](tags/{r['파일']}) |")
    lines += ["", "## DEAD — 0 아닌 값이 희소한 태그", "",
              "| 태그 | 설명 | 0아닌값 | 주의 |", "|---|---|---|---|"]
    for _, r in dead.iterrows():
        lines.append(f"| `{r['태그']}` | {r['설명']} | {r['0아닌값']:,} | {r['주의']} |")
    (config.OUTPUT_DIR / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    log.info("=== 완료: LIVE %d · DEAD %d → %s ===", len(live), len(dead), config.OUTPUT_DIR)
    pd.set_option("display.width", 220)
    print("\n=== 설비별 태그 수 ===")
    print(idx.groupby(["설비", "상태"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
