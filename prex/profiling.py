"""[진단] 전처리 산출물 기본 프로파일링 — 미리보기·차원·타입·통계량·클래스분포·상관·왜도.

모델링 착수 전 "데이터가 어떻게 생겼는지"를 표로 한 번에 확인하는 용도다.
`trend/`는 같은 내용을 **그림(PNG)** 으로 보여주지만 수치를 직접 읽기 어렵고 일부 항목
(dtypes·왜도·클래스 불균형비)은 아예 없다. 이 모듈은 **텍스트/CSV**로 낸다.

수행 항목
  1. 미리보기       head / tail
  2. 차원           shape + 메모리 사용량
  3. 데이터 타입    dtype + 결측수·결측률 + 고유값수
  4. 통계량         count·mean·std·min·p1·p25·p50·p75·p99·max
  5. 클래스 분포    알람 category / 태그별 이벤트 / 버너 ON·OFF / PI-D2P era  (+ 불균형비)
  6. 상관관계       Pearson + Spearman (왜도가 커서 둘을 같이 본다)
  7. 왜도           skewness + 첨도(kurtosis) + 해석 플래그

실행:  .venv/bin/python -m prex.profiling            (콘솔 출력 + CSV 저장)
       .venv/bin/python -m prex.profiling --quiet    (CSV만)

출력:  prex/output/profile/*.csv   (git 미포함)

⚠ 결측 처리: 상관행렬은 **pairwise(쌍별) 결측 제거**를 쓴다. `htx_eps`가 64%만 유효해서
   행 단위 dropna를 하면 표본이 통째로 날아간다. 대신 쌍마다 표본수가 달라지므로
   `corr_pearson_n.csv`(쌍별 유효표본수)를 같이 저장해 함께 볼 것.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

PROFILE_DIR = config.OUTPUT_DIR / "profile"
PREVIEW_ROWS = 5           # head/tail 행 수
SPEARMAN_SAMPLE = 200_000  # Spearman은 순위계산이 무거워 표본추출 (재현 위해 seed 고정)
SPEARMAN_SEED = 0
SKEW_STRONG = 1.0          # |skew| 가 이보다 크면 "강한 왜도" — 로그/분위수 변환 검토 대상


# --- 로드 -------------------------------------------------------------------
def load_outputs() -> dict[str, pd.DataFrame]:
    """prex 산출물 3종을 읽는다. 없으면 어떤 명령으로 만드는지 알려준다."""
    frames = {}
    for name in ("trend_htr31p", "alarm_events_htr31p", "fault_events_htr31p"):
        path = config.OUTPUT_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} 없음 — 먼저 `python -m prex.pipeline` 실행 필요.")
        frames[name] = pd.read_parquet(path)
    return frames


# --- 1~2. 미리보기 · 차원 -----------------------------------------------------
def preview(df: pd.DataFrame, name: str, rows: int = PREVIEW_ROWS) -> str:
    mem = df.memory_usage(deep=True).sum() / 1024**2
    head_cols = df.columns[:8]  # 콘솔 폭 때문에 앞 8개만 (전체는 dtypes 표에서 확인)
    return (
        f"\n{'='*100}\n[{name}]  shape = {df.shape[0]:,}행 × {df.shape[1]}열   메모리 {mem:,.1f} MB\n{'='*100}\n"
        f"--- head({rows}) · 앞 8열 ---\n{df[head_cols].head(rows).to_string()}\n"
        f"--- tail({rows}) · 앞 8열 ---\n{df[head_cols].tail(rows).to_string()}"
    )


# --- 3. 데이터 타입 + 결측 ----------------------------------------------------
def dtype_table(df: pd.DataFrame) -> pd.DataFrame:
    """열별 dtype·결측수·결측률·고유값수. 결측률은 '전체 행' 기준이다."""
    n = len(df)
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null": df.isna().sum(),
        "null_pct": (df.isna().sum() / n * 100).round(2),
        "nunique": df.nunique(dropna=True),
    })


# --- 4. 통계량 ----------------------------------------------------------------
def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """수치열 요약. p1/p99를 포함한다 — 계기 dropout 같은 극단치에 강한 범위를 보려고."""
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    return desc.rename(columns={"1%": "p1", "25%": "p25", "50%": "p50", "75%": "p75", "99%": "p99"})


# --- 5. 클래스 분포 -----------------------------------------------------------
def class_balance(series: pd.Series, label: str) -> pd.DataFrame:
    """범주형 분포 + 비율 + 불균형비(최다/최소). 라벨 희소성 판단용."""
    vc = series.value_counts(dropna=False)
    out = pd.DataFrame({"class": vc.index.astype(str), "count": vc.to_numpy()})
    out["pct"] = (out["count"] / out["count"].sum() * 100).round(3)
    out.insert(0, "field", label)
    return out


def collect_class_balance(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """이 데이터셋에서 '클래스'에 해당하는 것들을 한 표로 모은다."""
    trend, alarm, fault = (frames[k] for k in
                           ("trend_htr31p", "alarm_events_htr31p", "fault_events_htr31p"))
    parts = [
        class_balance(fault["category"], "fault_events.category"),
        class_balance(alarm["category"], "alarm_events.category"),
        class_balance(fault["tag"], "fault_events.tag"),
        class_balance(trend["PI-D2P_era"], "trend.PI-D2P_era"),
        class_balance(trend["htx_eps"].notna().map({True: "유효", False: "결측"}), "trend.htx_eps 유효성"),
    ]
    # 버너 ON/OFF — H31POH 이벤트를 1분 그리드에 전파해서 만든다 (verify_ua와 동일 방식)
    try:
        from .verify_ua import BURNER_ON_VALUE, burner_state_on_grid
        burner = burner_state_on_grid(alarm, trend.set_index("Time").index)
        state = burner.map(lambda v: "unknown" if pd.isna(v)
                           else ("ON" if v == BURNER_ON_VALUE else "OFF"))
        parts.append(class_balance(state, "trend.burner(H31POH)"))
    except Exception as exc:  # 이벤트가 없거나 형식이 다르면 이 항목만 건너뛴다
        logger.warning("버너 ON/OFF 분포 생략: %s", exc)
    return pd.concat(parts, ignore_index=True)


def imbalance_summary(balance: pd.DataFrame) -> pd.DataFrame:
    """필드별 불균형비 = 최다 클래스 / 최소 클래스. 클수록 단순 이진분류에 불리하다."""
    rows = []
    for field, g in balance.groupby("field", sort=False):
        hi, lo = g["count"].max(), g["count"].min()
        rows.append({"field": field, "n_class": len(g), "max": hi, "min": lo,
                     "imbalance_ratio": round(hi / lo, 1) if lo else np.inf})
    return pd.DataFrame(rows)


# --- 6. 상관관계 --------------------------------------------------------------
def correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pearson(전체, pairwise) · Pearson 쌍별 표본수 · Spearman(표본추출).

    Spearman을 같이 내는 이유: 이 데이터는 왜도가 커서(§7) 선형상관만 보면 오독한다.
    """
    num = df.select_dtypes(include=[np.number]).drop(columns=["segment_id"], errors="ignore")
    pearson = num.corr(method="pearson")           # pairwise 결측 제거가 기본 동작
    counts = num.notna().astype("int8")
    pair_n = pd.DataFrame(counts.T.to_numpy() @ counts.to_numpy(),
                          index=num.columns, columns=num.columns)
    sample = num.sample(n=min(SPEARMAN_SAMPLE, len(num)), random_state=SPEARMAN_SEED)
    spearman = sample.corr(method="spearman")
    return pearson.round(3), pair_n, spearman.round(3)


# --- 7. 왜도 ------------------------------------------------------------------
def skewness(df: pd.DataFrame) -> pd.DataFrame:
    """왜도·첨도와 해석 플래그. |skew|>SKEW_STRONG 이면 변환 검토 대상으로 표시한다."""
    num = df.select_dtypes(include=[np.number]).drop(columns=["segment_id"], errors="ignore")
    sk, ku = num.skew(), num.kurtosis()
    out = pd.DataFrame({"skew": sk.round(3), "kurtosis": ku.round(3)})
    out["방향"] = np.where(out["skew"] > 0, "오른쪽 꼬리", np.where(out["skew"] < 0, "왼쪽 꼬리", "대칭"))
    out["판정"] = np.where(out["skew"].abs() > SKEW_STRONG, "강한 왜도 → 변환 검토", "허용")
    return out.sort_values("skew", key=abs, ascending=False)


# --- 실행 ---------------------------------------------------------------------
def run(quiet: bool = False) -> dict[str, pd.DataFrame]:
    frames = load_outputs()
    trend = frames["trend_htr31p"]
    results: dict[str, pd.DataFrame] = {}

    if not quiet:
        for name, df in frames.items():
            print(preview(df, name))

    for name, df in frames.items():
        key = f"dtypes_{name}"
        results[key] = dtype_table(df)
        if not quiet:
            print(f"\n--- [{name}] 데이터 타입 · 결측 ---\n{results[key].to_string()}")

    results["describe_trend"] = describe_numeric(trend)
    if not quiet:
        print(f"\n--- [trend] 통계량 ---\n{results['describe_trend'].round(3).to_string()}")

    balance = collect_class_balance(frames)
    results["class_balance"] = balance
    results["class_imbalance"] = imbalance_summary(balance)
    if not quiet:
        print("\n--- 클래스 분포 (태그별 상위 10개만 표시) ---")
        for field, g in balance.groupby("field", sort=False):
            print(f"\n[{field}]")
            print(g.drop(columns="field").head(10).to_string(index=False))
        print(f"\n--- 클래스 불균형비 ---\n{results['class_imbalance'].to_string(index=False)}")

    pearson, pair_n, spearman = correlations(trend)
    results["corr_pearson"] = pearson
    results["corr_pearson_n"] = pair_n
    results["corr_spearman"] = spearman
    if not quiet:
        print(f"\n--- [trend] 상관행렬 Pearson (pairwise) ---\n{pearson.to_string()}")
        print(f"\n--- [trend] 상관행렬 Spearman (표본 {min(SPEARMAN_SAMPLE, len(trend)):,}행) ---\n{spearman.to_string()}")

    results["skewness_trend"] = skewness(trend)
    if not quiet:
        print(f"\n--- [trend] 왜도 · 첨도 ---\n{results['skewness_trend'].to_string()}")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    for key, df in results.items():
        df.to_csv(PROFILE_DIR / f"{key}.csv", encoding="utf-8-sig")
    logger.info("프로파일 CSV %d개 저장: %s", len(results), PROFILE_DIR)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="prex 산출물 기본 프로파일링")
    parser.add_argument("--quiet", action="store_true", help="콘솔 출력 없이 CSV만 저장")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    run(quiet=args.quiet)


if __name__ == "__main__":
    main()
