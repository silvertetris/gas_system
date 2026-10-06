"""계열 M 빙결 예측에 쓰인 원형 태그 12개 — 태그마다 시간 vs 값 그림 1장.

원형 = 가공 전 계측값. 1분(DI 는 상태변화) 원본을 일 단위로만 묶는다:
아날로그는 일평균 선 + 일 최저~최고 띠, 버너(DI)는 일 점화 비율[%].
학습에 쓰지 않은 구간(헤더온도 유효 전, 2013-10-01 이전)은 회색 음영.

    PYTHONPATH=. .venv/bin/python poster/raw_data_M.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pinn.restricted import config as r_config
from prex_multi import config as m_config

OUT_DIR = r_config.PROJECT_ROOT / "ttttt" / "M계열_원형데이터"

# 태그, 그림 제목(영어), y축 이름, 단위, 출처
TAGS = [
    ("TI61M", "Supply temperature to customer (target)", "Temperature", "°C", "raw"),
    ("PI61M", "Metering outlet pressure", "Pressure", "MPa", "raw"),
    ("PI21X", "Main line inlet pressure", "Pressure", "MPa", "raw"),
    ("TI21Y", "Heater inlet header temperature", "Temperature", "°C", "trend"),
    ("TSL_TSH_TSHH-31M", "Heater outlet header temperature", "Temperature", "°C", "trend"),
    ("TI33A", "Heater A outlet temperature", "Temperature", "°C", "trend"),
    ("TI33B", "Heater B outlet temperature", "Temperature", "°C", "trend"),
    ("TI-D2A", "Heater A water bath temperature", "Temperature", "°C", "trend"),
    ("TI-D2B", "Heater B water bath temperature", "Temperature", "°C", "trend"),
    ("ZI31D", "Bypass control valve opening", "Opening", "%", "trend"),
    ("H31AOH", "Heater A burner firing (daily ON ratio)", "ON ratio", "%", "burner_A"),
    ("H31BOH", "Heater B burner firing (daily ON ratio)", "ON ratio", "%", "burner_B"),
]


def load_daily() -> dict[str, pd.DataFrame]:
    trend_cols = [t for t, *_, s in TAGS if s == "trend"] + [s for *_, s in TAGS if s.startswith("burner")]
    g = pd.read_parquet(m_config.OUTPUT_DIR / "trend_all.parquet",
                        columns=["Time", *trend_cols]).set_index("Time").sort_index()
    # TI61M·PI61M·PI21X 는 prex_multi 산출물에 없다 → 원본 1시간 집계 캐시(시간평균)를 쓴다
    raw = pd.read_parquet(r_config.RAW_CACHE)
    out = {}
    for tag, *_, src in TAGS:
        if src == "raw":
            s = raw[tag]
        elif src == "trend":
            s = g[tag]
        else:
            s = g[src].astype("float") * 100.0
        d = s.resample("1D").agg(["mean", "min", "max"])
        if tag == "TI61M":      # 일 최저는 분 단위 최저로 (빙결 타깃의 원형)
            d["min"] = raw["TI61M_min"].resample("1D").min()
        if src.startswith("burner"):
            d[["min", "max"]] = float("nan")
        out[tag] = d
    return out


def plot(tag: str, title: str, yname: str, unit: str, d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
    fig.patch.set_facecolor("white")
    if d["min"].notna().any():
        ax.fill_between(d.index, d["min"], d["max"], color="#9bb7d4", alpha=0.45,
                        linewidth=0, label="daily min–max")
    ax.plot(d.index, d["mean"], color="#1f4e79", linewidth=0.7,
            label="daily ON ratio" if unit == "%" and "ON" in yname else "daily mean")
    if tag == "TI61M":
        ax.axhline(r_config.FREEZE_C, color="#c0392b", linewidth=0.9, linestyle="--",
                   label="freezing threshold 0 °C")
    ax.axvspan(d.index.min(), pd.Timestamp(r_config.VALID_FROM), color="0.85", zorder=0,
               label="not used for training")
    ax.set_title(f"{tag} — {title}", fontsize=10, loc="left")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{yname} [{unit}]")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(d.index.min(), d.index.max())
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7, loc="upper left", ncol=4, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = load_daily()
    for tag, title, yname, unit, _ in TAGS:
        plot(tag, title, yname, unit, daily[tag])
        d = daily[tag]
        print(f"{tag:18s} {d['mean'].first_valid_index():%Y-%m-%d}~{d['mean'].last_valid_index():%Y-%m-%d} "
              f"mean {d['mean'].mean():8.3f}  p1 {d['mean'].quantile(.01):8.3f}  p99 {d['mean'].quantile(.99):8.3f}  "
              f"min {d['min'].min():8.3f}  max {d['max'].max():8.3f}")


if __name__ == "__main__":
    main()
