"""열화 추적 — `U·A` 를 열수지 유량으로 직접 구해 추세와 정비 회복을 검정한다.

## 왜 이번엔 다른가

문서 07 §3 의 24개 검정은 전부 실패했다. 그때 `U·A` 는 **부서진 연쇄**에서 나왔다:
유량을 몰라 `ε`·`NTU` 로 대신했고, 그 `ε` 이 유량 정보를 담고 있지 않다는 것이 문서 10 에서
밝혀졌다. 이제 유량이 **수조 열수지**로 독립적으로 나온다(문서 12). 그래서

    U·A = Q_gas / ΔT_lm          ← ε-NTU 를 **전혀 거치지 않는다**
    ΔT_lm = (ΔT₁ − ΔT₂)/ln(ΔT₁/ΔT₂),  ΔT₁ = T_bath − T_in,  ΔT₂ = T_bath − T_out

로 정의대로 구할 수 있다. 유량 의존은 제작도서 직렬저항으로 정규화한다:

    U·A_정규 = U·A / [U(x)/U(설계)],    x = m/m_설계

오염이 쌓이면 `U·A_정규` 가 **떨어지고**, 정비하면 **회복**해야 한다. 그 둘을 검정한다.

## 검정 설계 (앞선 실패에서 배운 것)

| 항목 | 내용 |
|---|---|
| 집계 | **일별 중앙** — 사이클 단위는 자기상관이 심하다 |
| 추세 | Spearman(U·A_정규, 시간). CI 는 **월 블록 부트스트랩** |
| 회복 | 정비 전 30일 vs 후 30일. 8개 사건에 대한 **부호검정** |
| 대조군 | ① 시간 치환 ② **무작위 날짜**를 정비로 가정 (같은 개수·같은 창) |
| 교란 | 계절성·부하 잔여의존을 같이 본다. 둘 다 크면 판정 보류 |

⚠ `ΔT_lm` 은 `T_out` 을 쓴다. 문서 12 §5-3 의 수조 성층화 가설이 맞으면 `T_out` 이 화력에
따라 편향되고, 화력은 계절을 탄다. **계절성 점검이 그래서 필수다.**
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from prex import config as p_config
from . import config

logger = logging.getLogger(__name__)
RNG = np.random.default_rng(42)

# 문서 05 §2 — 데이터로 확정된 정지 구간
MAINTENANCE = ["2011-10-06", "2013-03-19", "2015-03-09", "2017-03-24",
               "2019-03-18", "2021-03-29", "2022-10-17", "2024-10-17"]
WINDOW_D = 60            # 정비 전/후 비교 창 [일]
GUARD_D = 3              # 정비 전후로 버릴 날 (작업 중 데이터 오염)
MIN_DAYS = 5             # 창 안 최소 유효 일수 (일별 집계가 드문 히터가 있다)
N_NULL = 2000            # 대조군 반복

A_IO, A_O = p_config.HTR_ALPHA_IO_KCAL, p_config.HTR_ALPHA_O_KCAL
EXP = p_config.GAS_SIDE_EXPONENT
R_FOUL = 1.0 / p_config.HTR_U_KCAL - (1.0 / A_IO + 1.0 / A_O)


def u_ratio(x: np.ndarray) -> np.ndarray:
    """U(x)/U(설계) — 가스측만 Dittus-Boelter 로 변한다."""
    r_ref = 1.0 / A_IO + 1.0 / A_O + R_FOUL
    return r_ref / (1.0 / (A_IO * np.clip(x, 1e-3, None) ** EXP) + 1.0 / A_O + R_FOUL)


def build() -> pd.DataFrame:
    """사이클별 `U·A` 를 만들고 일별 중앙으로 집계한다."""
    cyc = pd.read_parquet(config.OUTPUT_DIR / "heatflow_cycles.parquet")
    need = ["Time"] + [f"TI-D2{u}" for u in config.UNITS] + [f"TI33{u}" for u in config.UNITS] \
           + [config.TRAIN_INLET[t] for t in config.TRAIN_UNITS]
    grid = pd.read_parquet(config.OUTPUT_DIR / "trend_all.parquet",
                           columns=sorted(set(need))).set_index("Time").sort_index()

    rows = []
    for (train, unit), g in cyc.groupby(["계열", "히터"]):
        t_in = grid[config.TRAIN_INLET[train]]
        bath, out = grid[f"TI-D2{unit}"], grid[f"TI33{unit}"]
        # 사이클 구간 평균 온도 (t0~t1)
        i0 = grid.index.searchsorted(g["t0"].to_numpy())
        i1 = grid.index.searchsorted(g["t1"].to_numpy(), side="right")
        tb = np.array([bath.iloc[a:b].mean() for a, b in zip(i0, i1)])
        to = np.array([out.iloc[a:b].mean() for a, b in zip(i0, i1)])
        ti = np.array([t_in.iloc[a:b].mean() for a, b in zip(i0, i1)])
        d1, d2 = tb - ti, tb - to
        ok = (d1 > 1.0) & (d2 > 1.0) & np.isfinite(d1) & np.isfinite(d2)
        lmtd = np.where(ok, (d1 - d2) / np.log(np.where(ok, d1 / d2, np.e)), np.nan)
        ua = g["q_gas_kw"].to_numpy() * 1e3 / lmtd                     # W/K
        x = g["설계비"].to_numpy()
        rows.append(pd.DataFrame({
            "t0": g["t0"].to_numpy(), "계열": train, "히터": unit,
            "ua": ua, "ua_norm": ua / u_ratio(x), "x": x, "lmtd": lmtd,
            "q_gas_kw": g["q_gas_kw"].to_numpy(), "bath": tb, "t_out": to, "t_in": ti}))
    df = pd.concat(rows, ignore_index=True)
    df = df[np.isfinite(df["ua_norm"]) & (df["ua_norm"] > 0)]
    # 상하위 1% 절사 (히터별) — 사이클 추정 잡음
    keep = df.groupby("히터")["ua_norm"].transform(
        lambda s: s.between(s.quantile(0.01), s.quantile(0.99)))
    df = df[keep]
    daily = (df.set_index("t0").groupby(["히터", pd.Grouper(freq="1D")])
               .agg(ua_norm=("ua_norm", "median"), ua=("ua", "median"), x=("x", "median"),
                    n=("ua_norm", "size")).reset_index())
    daily = daily[daily["n"] >= 3]
    return df, daily


def _block_boot(y: np.ndarray, t: np.ndarray, dates: pd.DatetimeIndex, n=N_NULL):
    """월 블록 부트스트랩 Spearman CI — 자기상관을 감안한다."""
    key = dates.to_period("M")
    blocks = [np.flatnonzero(key == k) for k in key.unique()]
    out = []
    for _ in range(n):
        pick = RNG.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        if len(idx) > 20:
            out.append(stats.spearmanr(t[idx], y[idx])[0])
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def _design(g: pd.DataFrame, t0: pd.Timestamp):
    """설계행렬 — 부하(log m 의 자연스플라인)와 계절(월 조화 2차)을 명시적으로 뺀다.

    ⚠ 제작도서 `u_ratio` 만으로는 부하가 안 빠진다. `U·A = Q_gas/ΔT_lm = m·c_p·NTU` 인데
      이 설비의 `NTU` 는 유량에 거의 반응하지 않아(문서 10) `U·A ∝ m` 이 된다.
      정규화 후에도 ρ(부하) 0.84~0.92 로 남았다. 그래서 **경험적으로 회귀 제거**한다.
    부하는 계절을 타므로(겨울 대유량) 둘을 같이 넣어야 교락이 풀린다.
    """
    lm = np.log(np.clip(g["x"].to_numpy(), 1e-3, None))
    kn = np.quantile(lm, [0.2, 0.4, 0.6, 0.8])
    cols = [np.ones(len(g)), lm, lm ** 2]
    cols += [np.clip(lm - k, 0, None) ** 3 for k in kn]          # 절단 3차 스플라인
    doy = pd.DatetimeIndex(g["t0"]).dayofyear.to_numpy() / 365.25
    for h in (1, 2):
        cols += [np.sin(2 * np.pi * h * doy), np.cos(2 * np.pi * h * doy)]
    yrs = (pd.DatetimeIndex(g["t0"]) - t0).days.to_numpy() / 365.25
    return np.column_stack(cols), yrs


def trend_test(daily: pd.DataFrame) -> pd.DataFrame:
    """부하·계절을 뺀 뒤 남는 시간 추세가 열화 후보다. 시간 치환 대조군 동반."""
    rows = []
    for u, g in daily.groupby("히터"):
        g = g.sort_values("t0").reset_index(drop=True)
        t0 = g["t0"].min()
        X, yrs = _design(g, t0)
        y = np.log(g["ua_norm"].to_numpy())
        # 1단계: 부하·계절만으로 적합 → 잔차
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        r2_load = 1 - r.var() / y.var()
        # 2단계: 잔차의 연 추세
        A = np.column_stack([np.ones(len(g)), yrs])
        c, *_ = np.linalg.lstsq(A, r, rcond=None)
        slope = c[1]                                             # log/년
        rho, p = stats.spearmanr(yrs, r)
        lo, hi = _block_boot(r, yrs, pd.DatetimeIndex(g["t0"]))
        # 대조군: 월 블록을 섞어 시간축을 파괴
        key = pd.DatetimeIndex(g["t0"]).to_period("M")
        blocks = [np.flatnonzero(key == k) for k in key.unique()]
        null = []
        for _ in range(500):
            order = RNG.permutation(len(blocks))
            idx = np.concatenate([blocks[i] for i in order])
            null.append(np.polyfit(yrs, r[idx][:len(yrs)], 1)[0])
        rows.append({"히터": u, "일수": len(g),
                     "기간_년": round(float(yrs.max()), 1),
                     "부하계절_설명력R²": round(r2_load, 3),
                     "잔차 연변화%": round(100 * (np.exp(slope) - 1), 2),
                     "대조군 |연변화|p95%": round(100 * float(np.percentile(np.abs(np.exp(null) - 1), 95)), 2),
                     "ρ(잔차,시간)": round(rho, 3), "p": f"{p:.1e}",
                     "부트CI(ρ)": f"[{lo:+.3f}, {hi:+.3f}]",
                     "잔차ρ(부하)": round(stats.spearmanr(g["x"].to_numpy(), r)[0], 3),
                     "U·A중앙": round(float(np.median(g["ua_norm"])), 1)})
    return pd.DataFrame(rows)


def recovery_test(daily: pd.DataFrame) -> pd.DataFrame:
    """정비 전후 회복 — 부호검정 + 무작위 날짜 대조군."""
    rows = []
    for u, g in daily.groupby("히터"):
        g = g.sort_values("t0").reset_index(drop=True)
        if len(g) < 200:
            continue
        # 부하·계절을 뺀 **잔차**로 비교한다 (정비는 3월/10월에 몰려 있어 계절 교락이 크다)
        X, _ = _design(g, g["t0"].min())
        yv = np.log(g["ua_norm"].to_numpy())
        bb, *_ = np.linalg.lstsq(X, yv, rcond=None)
        s = pd.Series(yv - X @ bb, index=pd.DatetimeIndex(g["t0"])).sort_index()

        def delta(d0):
            d0 = pd.Timestamp(d0)
            pre = s.loc[d0 - pd.Timedelta(days=WINDOW_D + GUARD_D):d0 - pd.Timedelta(days=GUARD_D)]
            post = s.loc[d0 + pd.Timedelta(days=GUARD_D):d0 + pd.Timedelta(days=WINDOW_D + GUARD_D)]
            if len(pre) < MIN_DAYS or len(post) < MIN_DAYS:
                return np.nan
            return float(np.exp(post.median() - pre.median()) - 1.0)   # 로그잔차 → 비율

        obs = np.array([delta(d) for d in MAINTENANCE], dtype=float)
        ok = np.isfinite(obs)
        if ok.sum() < 3:
            rows.append({"히터": u, "유효사건": int(ok.sum()), "판정": "표본부족"})
            continue
        # 대조군: 같은 개수의 무작위 날짜
        lo, hi = s.index.min() + pd.Timedelta(days=60), s.index.max() - pd.Timedelta(days=60)
        span = (hi - lo).days
        null = []
        for _ in range(N_NULL):
            ds = lo + pd.to_timedelta(RNG.integers(0, span, ok.sum()), unit="D")
            v = np.array([delta(d) for d in ds], dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                null.append(np.median(v))
        null = np.array(null)
        med = float(np.nanmedian(obs[ok]))
        pval = float((null >= med).mean()) if len(null) else np.nan
        rows.append({"히터": u, "유효사건": int(ok.sum()),
                     "회복률중앙%": round(100 * med, 2),
                     "양수사건": f"{int((obs[ok] > 0).sum())}/{int(ok.sum())}",
                     "대조군중앙%": round(100 * float(np.median(null)), 2) if len(null) else np.nan,
                     "대조군p95%": round(100 * float(np.percentile(null, 95)), 2) if len(null) else np.nan,
                     "p(단측)": round(pval, 4),
                     "사건별%": " ".join("—" if not np.isfinite(v) else f"{100*v:+.0f}" for v in obs)})
    return pd.DataFrame(rows)


def sawtooth_test(daily: pd.DataFrame) -> pd.DataFrame:
    """열화의 결정적 형태 — **정비마다 초기화되는 톱니**인가.

    부하·계절을 뺀 잔차 `r` 에 대해 세 모형을 비교한다:
        M0  r = a                              변화 없음
        M1  r = a + b·t                        되돌아오지 않는 단조 추세 (센서 드리프트도 이렇다)
        M2  r = a_k + b·(t − t_k)              **정비마다 수준 초기화**, 기울기는 공통
    `M2 − M1` 개선이 열화+회복의 고유 서명이다. 단조 드리프트로는 설명되지 않는다.
    대조군: 정비일을 **무작위 날짜**로 바꿔 같은 M2 를 적합한다(같은 개수·같은 자유도).
    """
    rows = []
    for u, g in daily.groupby("히터"):
        g = g.sort_values("t0").reset_index(drop=True)
        if len(g) < 300:
            continue
        t0 = g["t0"].min()
        X, yrs = _design(g, t0)
        y = np.log(g["ua_norm"].to_numpy())
        bb, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ bb
        dates = pd.DatetimeIndex(g["t0"])

        def rss(cuts):
            """M2: 구간별 절편 + 공통 기울기."""
            seg = np.searchsorted(np.sort(np.array(cuts, dtype="datetime64[ns]")),
                                  dates.values, side="right")
            use = np.unique(seg)
            D = np.zeros((len(g), len(use) + 1))
            for j, sv in enumerate(use):
                D[seg == sv, j] = 1.0
            # 구간 시작으로부터의 경과 연수 (공통 기울기)
            el = np.zeros(len(g))
            for sv in use:
                m = seg == sv
                el[m] = yrs[m] - yrs[m].min()
            D[:, -1] = el
            c, *_ = np.linalg.lstsq(D, r, rcond=None)
            res = r - D @ c
            return float((res ** 2).sum()), float(c[-1]), len(use) + 1

        rss0 = float(((r - r.mean()) ** 2).sum())
        A1 = np.column_stack([np.ones(len(g)), yrs])
        c1, *_ = np.linalg.lstsq(A1, r, rcond=None)
        rss1 = float(((r - A1 @ c1) ** 2).sum())
        rss2, slope2, k2 = rss(MAINTENANCE)

        n = len(g)
        f = ((rss1 - rss2) / max(k2 - 2, 1)) / (rss2 / (n - k2))
        # 대조군: 무작위 날짜 (같은 개수)
        lo, hi = dates.min() + pd.Timedelta(days=120), dates.max() - pd.Timedelta(days=120)
        span = max((hi - lo).days, 1)
        null_f = []
        for _ in range(300):
            ds = lo + pd.to_timedelta(np.sort(RNG.integers(0, span, len(MAINTENANCE))), unit="D")
            rs, _, kk = rss(list(ds.astype(str)))
            null_f.append(((rss1 - rs) / max(kk - 2, 1)) / (rs / (n - kk)))
        null_f = np.array(null_f)
        rows.append({"히터": u, "일수": n,
                     "R²(M1 추세)": round(1 - rss1 / rss0, 4),
                     "R²(M2 톱니)": round(1 - rss2 / rss0, 4),
                     "톱니 기울기%/년": round(100 * (np.exp(slope2) - 1), 2),
                     "F(M2>M1)": round(f, 2),
                     "대조군 F 중앙": round(float(np.median(null_f)), 2),
                     "대조군 F p95": round(float(np.percentile(null_f, 95)), 2),
                     "p(단측)": round(float((null_f >= f).mean()), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cyc, daily = build()
    logger.info("사이클 %d → 일별 %d행", len(cyc), len(daily))
    cyc.to_parquet(config.OUTPUT_DIR / "degrade_cycles.parquet", index=False)
    daily.to_csv(config.OUTPUT_DIR / "degrade_daily.csv", index=False, encoding="utf-8-sig")
    tr = trend_test(daily)
    rc = recovery_test(daily)
    tr.to_csv(config.OUTPUT_DIR / "degrade_trend.csv", index=False, encoding="utf-8-sig")
    rc.to_csv(config.OUTPUT_DIR / "degrade_recovery.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 검정 1: U·A 추세 (열화면 ρ(시간) < 0) ===")
    print(tr.to_string(index=False))
    sw = sawtooth_test(daily)
    sw.to_csv(config.OUTPUT_DIR / "degrade_sawtooth.csv", index=False, encoding="utf-8-sig")
    print("\n=== 검정 2: 정비 회복 (열화면 회복률 > 0, 대조군보다 커야 한다) ===")
    print(rc.to_string(index=False))
    print("\n=== 검정 3: 톱니 (정비마다 초기화되는가 — 단조 드리프트와 구별) ===")
    print(sw.to_string(index=False))


if __name__ == "__main__":
    main()
