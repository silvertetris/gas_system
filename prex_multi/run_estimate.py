"""4대 히터 × 정비주기별 수조 열손실 Q_loss 추정 실행기.

**주기 정의** — 경계는 두 가지로 교차확인한 것만 쓴다:
  (a) 데이터 검출: 수조온도 <10℃ 가 24h 이상 지속(배수·냉각된 정지)
  (b) 정기점검 보고서 OCR (docs §9): 2021.03.29~04.02, 2024.10.17~10.30
  (b)의 두 날짜가 (a)의 P호기 검출목록과 분 단위로 일치한다 → 검출기 신뢰.

  그래서 1차 대상은 양끝이 모두 보고서로 확인된 **완전한 한 주기**:
      cycle-2021 : 2021-04-03 ~ 2024-10-16  (3.5년)
  비교용으로 현재 진행 중인 주기와 전체 구간도 같이 돌린다.

**m_gas 는 여기서 추정하지 않는다.** OFF 구간 열수지로는 식별 불가능함이 확인됐다
(사유는 estimate.py 의 "기각된 접근 2가지" 참고).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, estimate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CYCLES = [
    ("cycle-2021", "2021-04-03", "2024-10-16"),   # 양끝 정기점검 확인 — 완전한 한 주기
    ("cycle-2024", "2024-10-30", None),           # 현재 주기(진행 중)
    ("전체", None, None),
]


def main() -> None:
    tr, burner = estimate.load_all()
    log.info("trend %s  %s ~ %s", tr.shape, tr.index.min(), tr.index.max())

    # C_eff 는 주기와 무관한 설비 물성이므로 전체 구간에서 한 번만 추정한다.
    c_eff, ramps = {}, {}
    for u in config.UNITS:
        v, n = estimate.c_eff_from_ramps(tr, burner[u], u)
        c_eff[u], ramps[u] = v, n
        sens = config.scaled_mw_cw_j_per_k(u)
        log.info("C_eff[%s] = %.1f MJ/K  (승온 %d개, 현열추정 %.1f MJ/K 의 %.2f배)",
                 u, v / 1e6, n, sens / 1e6, v / sens)

    rows = []
    for name, lo, hi in CYCLES:
        sl = tr.loc[lo:hi] if (lo or hi) else tr
        for u in config.UNITS:
            runs = estimate.no_load_cooling(sl, burner[u].loc[sl.index], u)
            r = estimate.q_loss_kw(runs, c_eff[u])
            duty = config.design_duty_w(u) / 1e3
            rows.append({
                "cycle": name, "unit": u, "bath": config.BATH_TYPE[u],
                "from": sl.index.min().date(), "to": sl.index.max().date(),
                "C_eff_MJK": c_eff[u] / 1e6, "설계열량_kW": duty,
                "Q_loss_kW": r["q_loss_kw"], "q25": r["q25"], "q75": r["q75"],
                "손실률%": 100 * r["q_loss_kw"] / duty,
                "수조온도_C": r["T_bath"], "무부하런수": r["n"],
            })
            if np.isfinite(r["q_loss_kw"]):
                log.info("[%s] %s  Q_loss=%.1f kW (IQR %.1f~%.1f) = 설계의 %.1f%%  "
                         "수조 %.1f℃  런 %d개", name, u, r["q_loss_kw"], r["q25"],
                         r["q75"], 100 * r["q_loss_kw"] / duty, r["T_bath"], r["n"])
            else:
                log.info("[%s] %s  측정불가 — 무부하 런 %d개 (상시 통가스 계열)",
                         name, u, r["n"])

    out = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = config.OUTPUT_DIR / "q_loss.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    log.info("저장: %s", p)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
