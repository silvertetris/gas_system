# 포스터용 그림 모음 (2026-09-14)

원본 `pinn/restricted/output/optuna/<구조>_min/v_<정제>/` · `eda/output/tags/`. 분석용 다패널 그림 — 필요한 패널만 잘라 쓴다.
타깃: 6시간 안 분 단위 최저 공급온도 < 0℃ (센서 이상값 제거) · 시드 5 · 조합마다 Optuna 25

```
ttttt/
├── 1_학습중/     <모델>/<모델>_<정제>_<계열>_diag.png · _xai.png
├── 2_loss/       <모델>/<모델>_<정제>_<계열>_loss.png
├── 3_시험기간/   <모델>/<모델>_<정제>_<계열>_timeline.png · _ROC_PR.png
├── 4_태그추이/   <태그>.png
└── 5_모델비교/   model_timeline_<계열>_min_<정제>.png (모델 6개 시계열) · model_accuracy_min.csv/.md (정확도 표)
                  arch_compare_*.png · *_정제비교.png
```

- 모델 폴더: `soft_PINN`(물리=손실) · `soft_PINN_가중확장`(물리 가중 0.3~30) · `hard_PINN`(출력=물리식) · `nn`(물리 없음) · `LSTM`(시계열, 물리 없음)
- 정제: `KF` · `AE` · `EKF` · `앙상블_KF+AE+EKF` · `정제없음`
- 계열: `M`(HTR-31A/B, 도시가스) · `Z`(HTR-31O/P, 공항·발전)

**포스터용 영어 그림 (폴더 루트, 2026-09-15)** — 다이어그램은 투명 배경, 그래프는 흰 배경. 구성은 `docs/20_학회포스터_구성안.md` 2-3·2-5·3-3.

| 파일 | 내용 | 스크립트 |
|---|---|---|
| `Kalman_filter_diagram.png` | 칼만필터 개념도 (방법 1) | `poster/diagram_kalman.py` |
| `KF_softPINN_diagram.png` | KF–soft PINN 구조도, 박스 제목 + 화살표만 (방법 2) | `poster/diagram_kf_softpinn_en.py` |
| `Gas_system_diagram.png` | 가스 시스템 개략도 (연구 내용 1) | `poster/diagram_gas_system.py` |
| `KF_filtering_M.png` | KF 필터링 결과, 측정 vs 추정 48h (연구 내용 3) | `poster/kf_real_plot.py` |
| `PINN_loss_curve_M.png` | 최종 모델 학습 곡선 (연구 내용 4) | `poster/pinn_loss_plot.py` |
| `KF_softPINN_구조도.png` · `칼만필터_개념도.png` | 이전 한글판 (내부 참고) | — |

**`poster_graphs/` — 포스터에 넣을 그래프 전부 (영어 · 흰 배경 · 300 dpi · 한 그림 한 차트)** — 스크립트 `PYTHONPATH=. .venv/bin/python poster/poster_graphs.py`

| 파일 | 내용 | 섹션 |
|---|---|---|
| `01_KF_filtering_result.png` | KF 측정 vs 추정 48h | Experimental details |
| `02_PINN_loss_curve.png` | 최종 모델 학습·검증 손실 | Experimental details |
| `03_mu_JT_convergence.png` | 줄-톰슨 계수 학습 수렴 (문헌 0.56) | Experimental details |
| `04_optuna_history.png` | Optuna trial CV AUC · 누적 최고 | Experimental details |
| `05_fold_validation_auc.png` | fold 별 검증 AUC | Experimental details |
| `06_refinement_comparison.png` | 정제 5종 CV AUC(선정 기준) · 시험 AUC | Experimental details / Results |
| `07_model_comparison_AUC.png` | 모델 6종 시험 AUC ± SD | Results |
| `08_model_comparison_alarm_hits.png` | 상위 1% 경보가 잡은 빙결 시간 | Results |
| `09_test_period_timeline.png` | 시험구간 전체: 최저 공급온도 실제·예측 + 빙결 확률 | Results |
| `10_test_zoom_10days.png` | 빙결 최다 10일 확대 (±1σ) | Results |
| `11_monthly_alarm_hits.png` | 월별 빙결 시간 · 경보 적중 | Results |
| `12_ROC_PR_curves.png` | ROC · PR (시드 42 모델) | Results |
| `13_physics_attribution.png` | 물리 분해 (빙결 − 평시) | Results |
| `14_input_group_importance.png` | 입력 그룹 SHAP 비중 | Results |
| `15_top10_input_features.png` | 상위 10 입력 SHAP | Results |

| 파일 | 볼 패널 |
|---|---|
| `*_diag.png` (1_학습중) | ② 물리 모수 μ_JT 수렴(nn·LSTM 은 물리층 없음) · ③ Optuna 탐색 이력 · ④ fold 별 검증 AUC |
| `*_xai.png` | ① SHAP 상위 입력 · ② 그룹 기여 · ③ 그룹 섞기 AUC 하락 · ④ 물리 분해 · ⑤ fold 별 물리 모수 · ⑥ 하이퍼파라미터 중요도 |
| `*_loss.png` (2_loss) | ① 학습곡선 (학습/검증 손실, 시드 5, 로그축) |
| `*_timeline.png` | ① 공급온도 실제 vs 예측 (2023-08~2026-06) · ② 빙결 확률 모델 vs GBM · ③ 월별 경보 적중 · ④⑤ 10일 확대 |
| `*_ROC_PR.png` | ⑤ ROC · ⑥ PR · ⑦ 보정 곡선 · ⑧ 최저온 전후 3주 |
| `4_태그추이/<태그>.png` | 왼쪽 위 "전체 시계열" (2011~2026, 점선 = 정기점검) |

⚠ 실행 중이라 아직 없는 것: **LSTM × EKF (M·Z)**, 1시간 평균 타깃 전부. `arch_compare_*.png` 는 LSTM·가중확장 반영 전 그림.
