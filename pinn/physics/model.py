"""검정 모델 — 합류점을 예측하고 **가중방식만 바꾼 대조군**과 비교한다.

## 구조 (모수 10개 남짓. 신경망은 상한 대조군에만 쓴다)

    ε_u 실측  →  NTU_u = −ln(1−ε_u)                        ← 세 온도로 직접 계산
    x_u       =  F⁻¹( NTU_u / exp(ua_log[u]) )              ← 직렬저항 U(x) 의 역함수
    m_u       =  x_u · m_설계,u
    β         =  σ( a₀ + Σ_k softplus(c_k)·relu(v − v_k) )  ← 밸브개도의 **단조** 함수
    T_hdr     =  (1−β)·Σ(m_u·T_out,u)/Σm_u + β·T_in         ← (Q4)

`F(x) = NTU_설계 · U(x)/U(설계) / x` 는 치역 전체에서 단조감소라 역함수가 유일하다.
`ua_log` 는 히터당 1개 — **시불변**이다. 그래서 693만 행에 대해 과결정이고 반증 가능하다.

## 대조군 — 이게 핵심이다

물리 가중이 균등·설계 가중을 못 이기면 "열전달식이 합류점을 설명한다"고 말할 수 없다.

| 코드 | 가중 | 무엇을 검정하나 |
|---|---|---|
| `ntu_series` | ε-NTU 역산 (직렬 U, `ua_log` 학습) | 본 모델 |
| `ntu_flat` | ε-NTU 역산 (상수 U) | U(m) 의 형태가 중요한가 |
| `equal` | m_A = m_B | 유량 가중 자체가 필요한가 |
| `design` | m_u ∝ m_설계,u | 시변 유량이 필요한가 |
| `free` | 신경망이 가중을 직접 출력 | **상한** — 물리가 얼마나 손해인가 |
| `hotmax` | (1−β)·max(T_out) + β·T_in | 현행 `prex_multi` 근사 |
| `eps_shuffle` | ε 을 시간축에서 치환 후 역산 | **영가설 대조군** |
| `ntu_gated` | 역산 + `regime=isolated` 인 히터는 **유량 0** | ε≈0 의 2가 모호성이 원인인가 |
| `regime_gate` | m_u ∝ m_설계,u · 1[regime=flow] | ε 대신 **어느 히터가 켜졌나**만으로 되나 |
| `valve_only` | (1−β)·상수 + β·T_in — 상류온도를 **상수**로 | 히터 물리가 기여하는 몫이 있나 |
| `duty_weight` | m_u ∝ m_설계,u·(duty_u + f₀) | **버너가 켜진 히터가 사용 중인 히터**인가 |
| `eps_weight` | m_u ∝ m_설계,u·ε_u | ε 의 **부호를 뒤집으면** 맞나 (기술적 변형) |

### 매설 구간 (`_b` 조합, 문서 11 §2-2)

P&ID 상 각 히터 출구는 `TI-33x` **직후 지중 매설**되고, 합류는 그 뒤 매설 헤더에서 일어난다.
그러니 (Q4) 에 `TI-33x` 를 그대로 넣은 것은 **틀린 지점의 온도**다. 가지마다 먼저:

    T_합류,u = T_지중 + (T_33,u − T_지중)·exp(−N_u),   N_u = U·A_매설,u/(m_u·c_p) = k_u/m_u

매설 길이·심도 자료가 없으므로 `k_u` 를 히터당 **학습 모수 1개**로 두고, 식별된 값이
물리적으로 그럴듯한지 사후에 본다. 지중온도는 계절 조화항으로 둔다:

    T_지중 = g₀ + g₁·sin(2πdoy/365) + g₂·cos(2πdoy/365)

저유량 가지일수록 `N_u` 가 커져 지중온도로 무너진다 — 배관 구간에서 이미 확인한 구조다.

### ε 의 부호가 반대다

밸브 최소개도 구간에서 혼합식을 직접 풀면(β≈0 이므로 `w_A = (T_hdr−T_out,B)/(T_out,A−T_out,B)`)

    Spearman(w_A, ε_A − ε_B)    = +0.791 (M) / +0.530 (Z)   ← 식은 **음수**를 요구한다
    Spearman(w_A, duty_A − duty_B) = +0.601 (M) / +0.444 (Z)

**ε 이 높은 쪽이 유량이 많다.** ε-NTU 는 그 반대를 말한다. 원인은 ε≈0 의 해석이다 —
가동하지 않는 히터의 출구 열전대는 **정체 가스**를 읽어 입구온도 근처로 떨어진다
(ε→0). 역산은 그걸 "대유량"으로 읽는다. 이 설비에서 ε≈0 은 **무유량**이다.

### ε≈0 의 2가 모호성

`T_out ≈ T_in` 은 두 경우가 **모두** 만든다: (a) 유량이 매우 커서 데울 시간이 없다,
(b) 가스가 그 히터로 아예 가지 않는다(차단). 역산은 언제나 (a) 로 읽는다 — 차단된
히터에 **거대한 유량**을 주고, 그 찬 출구온도가 가중평균을 지배한다. `ntu_gated` 는
관측레이어의 `regime` 판정으로 (b) 를 골라내 유량 0 을 준다.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from . import config

WEIGHTINGS = ("ntu_series", "ntu_flat", "ntu_gated", "equal", "design", "free",
              "hotmax", "regime_gate", "valve_only", "duty_weight", "eps_weight",
              "eps_shuffle",
              # --- 매설 구간을 넣은 판 (문서 11 §2-2)
              "ntu_series_b", "eps_weight_b", "equal_b", "eps_shuffle_b")

# 매설 보정을 쓰는 조합. `_b` 접미사.
BURIED = {w for w in WEIGHTINGS if w.endswith("_b")}


def _u_ratio(x):
    """U(x)/U(설계) — 가스측만 Dittus-Boelter 로 변한다."""
    a_io, a_o, rf = config.ALPHA_IO_KCAL, config.ALPHA_O_KCAL, config.R_FOUL
    r_ref = 1.0 / a_io + 1.0 / a_o + rf
    return r_ref / (1.0 / (a_io * x ** config.GAS_SIDE_EXPONENT) + 1.0 / a_o + rf)


def _inv_grid(n: int = 6000):
    """F(x) = NTU_설계·U(x)/U(설계)/x 의 역함수 보간격자 (log-log, 증가순)."""
    x = np.geomspace(config.X_MIN, config.X_MAX, n)
    f = config.DESIGN_NTU * _u_ratio(x) / x
    assert np.all(np.diff(f) < 0), "F 가 단조감소가 아니다"
    return torch.tensor(np.log(f[::-1].copy())), torch.tensor(np.log(x[::-1].copy()))


def _interp(xq, xp, fp):
    """선형보간. `xq` 에 대해 미분 가능 (searchsorted 는 인덱스만 고른다)."""
    i = torch.searchsorted(xp, xq.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[i - 1], xp[i]
    y0, y1 = fp[i - 1], fp[i]
    t = ((xq - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return y0 + t * (y1 - y0)


class MixingTest(nn.Module):
    """(Q4) 검정기. `weighting` 만 바꿔 대조군을 만든다."""

    def __init__(self, weighting: str, units: list[str], n_feat: int,
                 valve_knots: np.ndarray):
        super().__init__()
        assert weighting in WEIGHTINGS, weighting
        self.weighting = weighting
        self.units = list(units)
        nu = len(units)

        # β(밸브) 단조맵 — 모든 대조군이 **동일 형태**로 각자 최적 적합한다 (공정 비교)
        self.beta_a = nn.Parameter(torch.tensor(0.0))
        self.beta_c = nn.Parameter(torch.full((len(valve_knots),), -5.0))
        self.register_buffer("valve_knots", torch.tensor(valve_knots, dtype=torch.float32))

        # U·A 보정 — 히터당 1개, 시불변
        self.ua_log = nn.Parameter(torch.zeros(nu))
        self.up_const = nn.Parameter(torch.tensor(35.0))     # valve_only 의 상류온도 상수
        self.duty_floor = nn.Parameter(torch.tensor(-2.0))   # duty_weight 의 바닥 유량
        self.register_buffer("m_d", torch.tensor([config.UNIT_MD[u] for u in units],
                                                 dtype=torch.float32))
        lf, lx = _inv_grid()
        self.register_buffer("grid_lf", lf.float())
        self.register_buffer("grid_lx", lx.float())

        # 매설 구간: N_u = exp(log_k[u]) / m_u,  지중온도는 계절 조화
        self.log_k = nn.Parameter(torch.full((nu,), 0.0))
        self.tg = nn.Parameter(torch.tensor([12.0, 0.0, -3.0]))

        if weighting == "free":
            self.net = nn.Sequential(nn.Linear(n_feat, config.HIDDEN), nn.SiLU(),
                                     nn.Linear(config.HIDDEN, config.HIDDEN), nn.SiLU(),
                                     nn.Linear(config.HIDDEN, nu))

    # ------------------------------------------------------------------ 물리
    def beta(self, valve):
        h = self.beta_a + (nn.functional.softplus(self.beta_c)
                           * torch.relu(valve.unsqueeze(-1) - self.valve_knots)).sum(-1)
        return torch.sigmoid(h)

    def flow_from_eps(self, eps, flat: bool):
        """ε → NTU → x → m. `flat=True` 면 U 를 상수로 본다(대조군)."""
        ntu = -torch.log((1.0 - eps).clamp_min(1e-6))
        if flat:
            x = (config.DESIGN_NTU / ntu).clamp(config.X_MIN, config.X_MAX)
        else:
            z = ntu * torch.exp(-self.ua_log)
            lz = torch.log(z).clamp(float(self.grid_lf[0]), float(self.grid_lf[-1]))
            x = torch.exp(_interp(lz, self.grid_lf, self.grid_lx))
        return x * self.m_d

    def ground_temp(self, doy_sin, doy_cos):
        return self.tg[0] + self.tg[1] * doy_sin + self.tg[2] * doy_cos

    def bury(self, t_out, m, doy_sin, doy_cos):
        """(Q4-b) 합류 전 매설 구간. T = T_g + (T_out − T_g)·exp(−k/m)."""
        tg = self.ground_temp(doy_sin, doy_cos).unsqueeze(-1)
        n = torch.exp(self.log_k) / m.clamp_min(1e-6)
        return tg + (t_out - tg) * torch.exp(-n.clamp(max=30.0))

    def forward(self, feat, eps, out, t_in, valve, gate=None, flow_gate=None, duty=None,
                doy_sin=None, doy_cos=None):
        b = self.beta(valve)
        if self.weighting == "valve_only":
            return (1.0 - b) * self.up_const + b * t_in
        if self.weighting == "hotmax":
            return (1.0 - b) * out.max(dim=-1).values + b * t_in

        if self.weighting in ("ntu_series", "eps_shuffle", "ntu_gated",
                              "ntu_series_b", "eps_shuffle_b"):
            m = self.flow_from_eps(eps, flat=False)
            if self.weighting == "ntu_gated":
                m = m * (1.0 - gate)              # 차단 히터는 유량 0
        elif self.weighting == "ntu_flat":
            m = self.flow_from_eps(eps, flat=True)
        elif self.weighting == "equal":
            m = torch.ones_like(out)
        elif self.weighting == "design":
            m = self.m_d.expand_as(out)
        elif self.weighting == "regime_gate":
            m = self.m_d.expand_as(out) * flow_gate
        elif self.weighting == "duty_weight":
            m = self.m_d * (duty + nn.functional.softplus(self.duty_floor))
        elif self.weighting in ("eps_weight", "eps_weight_b"):
            m = self.m_d * eps.clamp_min(1e-3)
        elif self.weighting == "equal_b":
            m = torch.ones_like(out)
        else:                                            # free
            m = nn.functional.softplus(self.net(feat)) + 1e-3

        w = m / m.sum(-1, keepdim=True).clamp_min(1e-9)
        # 매설 구간을 먼저 통과시킨 뒤 합류시킨다 (가중은 동일한 유량비)
        t_up = self.bury(out, m, doy_sin, doy_cos) if self.weighting in BURIED else out
        mix = (w * t_up).sum(-1)
        if self.weighting == "ntu_gated":
            # 두 히터 모두 차단이면 혼합온도가 정의되지 않는다 → 입구온도
            mix = torch.where(gate.sum(-1) >= len(self.units) - 0.5, t_in, mix)
        elif self.weighting == "regime_gate":
            # 통가스 히터가 없으면 혼합온도가 정의되지 않는다 → 뜨거운 쪽 출구
            mix = torch.where(flow_gate.sum(-1) < 0.5, out.max(dim=-1).values, mix)
        return (1.0 - b) * mix + b * t_in


def valve_knots(v: np.ndarray, k: int = config.BETA_KNOTS) -> np.ndarray:
    """밸브개도 분위 매듭. 값이 뭉쳐 있으면 중복을 제거한다."""
    q = np.unique(np.quantile(v, np.linspace(0.05, 0.95, k)))
    return q.astype(np.float32)


def n_params(m: MixingTest) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
