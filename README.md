# 영종 공급관리소 가스히터 고장예측 (KF-PINN)

한국가스공사 영종 공급관리소 가스히터 **4대(HTR-31A · 31B · 31O · 31P)** 의
열화지표를 추정하고 고장을 예측한다.

목표 3단계: **① Hard fault → ② Degradation → ③ RUL**

## 문서

분석 결과 문서는 **[docs/README.md](docs/README.md)** 에서 시작한다.
데이터 자체에 대한 1차 문서는 `data/docs/` (KPOS 태그사전 등).

## 코드 구조

디렉토리마다 `config.py` + 기능 모듈 + `pipeline.py`(진입점) 구성.
산출물은 `<dir>/output/` (git 미포함).

| 디렉토리 | 역할 | 실행 |
|---|---|---|
| `pid/` | P&ID `.dwg` → DXF/텍스트 판독 | `python -m pid.pipeline` |
| **`prex_multi/`** | **4대 히터 전처리 (현행)** — 1분 격자, 버너상태, 바이패스 β, 운전상태 | `python -m prex_multi.pipeline` |
| `prex/` | HTR-31P 1대 전용 **구버전** — 바이패스 미반영. `trend/` EDA 만 사용 | `python -m prex.pipeline` |
| `trend/` | EDA 시각화 (구버전 입력) | `python -m trend.pipeline` |
| `lstm/`, `autoencoder/`, `kalman/` | 파일럿 (원시 센서 기준, 물리층 이후 재설계 예정) | 각 `test/pipeline.py` |

### prex_multi 분석 모듈

| 모듈 | 역할 |
|---|---|
| `trend.py` | 전처리 본체 — 버너상태·β·운전상태 산출 |
| `verify_regime.py` | 관측 레이어 검증 T1~T4 |
| `crosscheck.py` | (H1) 수조 열수지로 (H5) 합류점 식 교차검증 |
| `backsolve.py` | **버너 가동률로 유량 역산** (유량계가 없으므로) |
| `estimate.py` | ⚠ 바이패스 이전 추정기 — **기각됨**. 기각 사유 기록용으로만 보존 |

## 환경

```bash
.venv/bin/python -m prex_multi.pipeline      # 전처리 (~40초, 7.8M행 × 52열)
```

GPU: RTX 4060 (torch 바로 동작, TF 는 `LD_LIBRARY_PATH` 필요).
P&ID 변환에 ODA File Converter + `xvfb-run` 필요.
