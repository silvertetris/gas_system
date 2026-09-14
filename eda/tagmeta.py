"""태그 메타데이터 — `data/docs/02_KPOS_태그사전.md` 를 파싱한다.

사전은 IO LIST(`251016 영종 IO LIST.xlsx`)에서 만들어졌고 **설비·부위·측정량·계기기능·역할**이
표로 정리돼 있다. 그림마다 이 정보를 실어야 "이게 무슨 설비의 무슨 태그인지" 알 수 있다.

⚠ 사전의 **"표만 보면 오해하는 태그"** 절에 실측으로 확인된 예외가 있다. 그것도 같이 싣는다:
  · `TSL_TSH_TSHH-31M/Z` — 표에는 온도 스위치지만 **실제로는 연속 아날로그**(헤더 온도)
  · `TI33C` — H-31C 는 **존재하지 않는 설비**(전 구간 0)
  · `RSF41D`/`RS41D` — 비상정압기라 상시 대기(전 구간 0)
  · `ZI41A~Q` — 스팬이 0~100 / 0~500 / 0~3000 혼재
  · `PI-D2P` — 2015·2025 스팬 변경
"""
from __future__ import annotations

import glob
import re
from functools import lru_cache

DICT_GLOB = "data/docs/*.md"

# 사전 §0 의 예외 (실측 확인). 그림 주석으로 싣는다.
CAVEATS = {
    "TSL_TSH_TSHH-31M": "⚠ 표에는 온도 스위치로 되어 있으나 실제는 연속 아날로그 — M계열 히터 출구 합류 헤더 온도. 2013-06부터 유효",
    "TSL_TSH_TSHH-31Z": "⚠ 표에는 온도 스위치로 되어 있으나 실제는 연속 아날로그 — Z계열 히터 출구 합류 헤더 온도. 2013-06부터 유효",
    "TI33C": "⚠ H-31C 는 존재하지 않는 설비(도면상 FUTURE 레이어). 전 구간 0 — 사용 불가",
    "RSF41D": "⚠ PCV-41D 는 비상정압기라 상시 대기. 전 구간 0 — 사용 불가",
    "RS41D": "⚠ PCV-41D 는 비상정압기라 상시 대기. 전 구간 0 — 사용 불가",
    "PI-D2P": "⚠ HTR-31P 진공식 히터의 핵심 상태변수. 2015·2025 스팬 변경 → 구간별 재정규화 필요",
    "PDI21A": "⚠ 분해능 0.05 · 중앙 0.00 — 유량/막힘 지표로 사용 불가",
    "PDI21B": "⚠ 분해능 0.05 · 중앙 0.00 — 유량/막힘 지표로 사용 불가",
    "PDI21O": "⚠ 분해능 0.05 · 중앙 0.00 — 유량/막힘 지표로 사용 불가",
    "PDI21P": "⚠ 분해능 0.05 · 중앙 0.00 — 유량/막힘 지표로 사용 불가",
}
for _t in ("ZI41A", "ZI41B", "ZI41C", "ZI41D", "ZI41O", "ZI41P", "ZI41Q"):
    CAVEATS[_t] = "⚠ 스팬이 시기별 0~100 / 0~500 / 0~3000 혼재. O/P/Q 는 중앙 100·p10 0 으로 사실상 개폐신호"
for _t in ("RS41A", "RS41B", "RS41C", "RS41O", "RS41P", "RS41Q"):
    CAVEATS[_t] = "⚠ RSF(설정압)의 스팬 백분율이라 중복 정보"


def _dict_path() -> str:
    cands = [p for p in glob.glob(DICT_GLOB) if "KPOS" in p]
    if not cands:
        raise FileNotFoundError("KPOS 태그사전을 찾을 수 없다")
    return cands[0]


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    """태그 → {설비, 절, 설명, 호기, 측정량, 계기기능, 역할, 신호, 배선, 상태}

    ⚠ 사전에는 **헤더가 세 종류**다. 위치를 고정하면 깨진다(1차 파서가 9열만 받아
      §3~§5 의 8열 표를 통째로 놓쳤고, LIVE 47개 중 9개가 "미등재"로 떨어졌다):
        9열 | 태그 | 설명 | 호기/라인 | 측정량 | 계기기능 | 역할 | 신호 | 배선 | 상태/이벤트 |
        8열 | 태그 | 설명 | 측정량 | 계기기능 | 역할 | 신호 | 배선 | 상태/이벤트 |
        4열 | 태그 | 표에는 | 실제 | 조치 |            ← §0 예외표. 본문표가 아니다
      → **헤더를 읽어 열 이름으로 매핑**한다.
    """
    text = open(_dict_path(), encoding="utf-8").read()
    out: dict[str, dict] = {}
    facility = section = ""
    header: list[str] | None = None
    want = {"설명", "측정량", "계기기능", "역할", "신호", "배선"}

    for line in text.splitlines():
        if line.startswith("## "):
            facility = re.sub(r"^##\s*\d+\.\s*", "", line).split("—")[0].strip()
            section, header = "", None
            continue
        if line.startswith("### "):
            section = re.sub(r"^###\s*[\d\-]+\.\s*", "", line).strip()
            header = None
            continue
        if not line.startswith("|"):
            header = None
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "태그":                     # 헤더 행
            header = cells if want.issubset(set(cells)) else None
            continue
        if header is None or re.fullmatch(r"[-: ]+", cells[0] or "-"):
            continue
        m = re.match(r"^`([^`]+)`", cells[0])
        if not m or "/" in m.group(1):                       # `PDI21A/B/O/P` 같은 묶음 표기 제외
            continue
        tag = m.group(1)
        row = dict(zip(header, cells))
        desc = re.sub(r"\*\((.*?)\)\*", r"\1", row.get("설명", "")).strip()
        out[tag] = {
            "설비": facility, "절": section, "설명": desc,
            "호기": row.get("호기/라인", ""),
            "측정량": row.get("측정량", ""), "계기기능": row.get("계기기능", ""),
            "역할": row.get("역할", ""), "신호": row.get("신호", ""),
            "배선": row.get("배선", ""), "상태": row.get("상태/이벤트", ""),
            "주의": CAVEATS.get(tag, ""),
        }
    return out


def describe(tag: str) -> dict:
    d = load().get(tag)
    if d:
        return d
    return {"설비": "미등재", "절": "", "설명": "(태그사전에 없음)", "호기": "",
            "측정량": "", "계기기능": "", "역할": "", "신호": "", "배선": "",
            "상태": "", "주의": CAVEATS.get(tag, "")}


if __name__ == "__main__":
    d = load()
    print(f"태그 {len(d)}개 파싱")
    for t in ("TI21Z", "TI-D2P", "TSL_TSH_TSHH-31Z", "ZI31Q", "LI-D1A", "TI61Z"):
        v = describe(t)
        print(f"\n{t}: {v['설비']} / {v['절']}")
        print(f"   {v['설명']} | {v['측정량']} | {v['역할']} | {v['상태']}")
        if v["주의"]:
            print(f"   {v['주의']}")
