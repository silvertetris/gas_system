"""PINN 모델 — 히터 블록은 **ε 을 관측값으로** 쓴다 (2026-09-13 개정, 문서 10 반영).

## 물리 연쇄

    (Q3) 출구온도    T_out,u = T_in + ε_u·(T_bath,u − T_in)          ε_u 는 **관측 가능**
    (Q4) 합류점      T_hdr   = (1−β)·Σ w_u·T_out,u/Σw_u + β·T_in
                     w_u     = m_설계,u · ε_u                        ← 경험 가중 (문서 10)
                     β       = σ(a + Σ softplus(c_k)·relu(v − v_k))  ← 밸브개도 단조맵
    (Q5) 줄-톰슨     T_reg   = T_hdr − μ_JT·ΔP
    (Q6) 매설배관    T_61    = T_g + (T_reg − T_g)·exp(−N),  T_g = a + b·T_in

## 🔴 왜 ε-NTU 유량 역산을 뺐나 (문서 10)

이전 판은 신경망이 유량비 `x_u` 를 내고 `ε_u = 1 − exp(−U·A(x)/(x·m_d·c_p))` 로 역산했다.
**그 방향이 틀렸다.** 1분 원해상도 전기간·대조군 12개 홀드아웃 검정 결과:

| 가중 | M 홀드아웃 R² | Z 홀드아웃 R² |
|---|---|---|
| `w_u ∝ m_d,u·ε_u` (채택) | **+0.522** | +0.327 |
| 영가설 (ε 시간치환) | −0.018 | −0.049 |
| ε-NTU 역산 (구 판) | **−2.712** | **−1.907** |

밸브 최소개도 구간에서 혼합비를 직접 풀면 `Spearman(w_A, ε-NTU 역산 w_A) = −0.735/−0.547`
(대조군 ±0.002) 이고 `Spearman(w_A, ε_A−ε_B) = +0.791/+0.530` 이다 — **부호가 반대다.**

원인: 정지한 히터의 출구 열전대가 **정체 가스**를 읽어 `ε→0` 이 된다. 역산은 그걸 항상
"대유량"으로 읽고 정지 히터에 거대한 유량을 준다. 이 설비에서 `ε≈0` 은 **무유량**이다.

그래서 `w_u ∝ ε_u` 는 열전달식이 아니라 **가동 상태의 연속 대리**로 읽어야 한다.
`U·A` 와 `m` 의 분리는 포기한다(문서 07 §2 유지). 대신 **(Q3)·(Q4) 의 모든 입력이
관측 가능해져** 전부 지도 학습된다 — `ε_u`·`T_bath,u`·`T_in`·`valve`·`ΔP`.

## 분산 전파 — 몬테카를로

`exp(−N)` 이 비선형이라 가우시안 전파가 부정확하다. 미래 상태 분포에서 **재매개화 표본**을
뽑아 물리층을 통과시키고 경험 평균·분산을 쓴다(미분 가능).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from . import config

KCAL_M2HC_TO_W = 4186.8 / 3600.0        # kcal/m²h℃ → W/m²K (비는 무차원이라 소거되지만 명시)


class FreezePINN(nn.Module):
    """상태 순서 [t_in, dp, valve, bath_u..., eps_u...] — 전부 관측 가능하다."""

    def __init__(self, n_feat: int, units: list[str], valve_knots, 
                 hidden: int = config.HIDDEN):
        super().__init__()
        self.units = list(units)
        nu = len(self.units)
        self.n_state = 3 + 2 * nu

        self.trunk = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.state = nn.Linear(hidden, 2 * self.n_state)     # (μ, logσ) × 상태
        # 유량 — **설계 대비 비 x_u = m_u/m_d,u** 를 유닛별로 직접 낸다 (+ 바이패스 1개).
        # ⚠ softmax 배분 + 총유량 log 조합은 **실패했다**(5차 버그). `ε(m)` 이 저유량에서
        #    포화(ε→1, 기울기 0)하는데 softmax 가 배분을 0 으로 밀어 그 구석에 갇혔다
        #    (m 0.1 kg/s, β 0.0, Q4 오차 6℃). 유효한 해가 범위 안에 있는데도 못 찾았다.
        #    x_u 를 직접 내면 ε 의 stiff 한 부분과 혼합비 β 가 **독립 노브**가 된다.
        # β(밸브) 단조맵 — 독립 센서(바이패스 밸브 개도)로 바이패스 분율을 구동한다.
        # ⚠ 자유 헤드로 두면 0.05 로 붕괴한다(실측 0.48/0.64). 문서 10 §3.
        self.beta_a = nn.Parameter(torch.tensor(0.0))
        self.beta_c = nn.Parameter(torch.full((len(valve_knots),), -5.0))
        self.register_buffer("valve_knots",
                             torch.as_tensor(valve_knots, dtype=torch.float32))
        self.ntu_pipe = nn.Linear(hidden, 1)

        # --- 전역 물리 모수
        lo, hi = config.MU_JT_MIN, config.MU_JT_MAX
        p0 = (config.MU_JT_INIT - lo) / (hi - lo)
        self.mu_jt_raw = nn.Parameter(torch.tensor(math.log(p0 / (1 - p0))))
        self.tg_a = nn.Parameter(torch.tensor(0.0))
        self.tg_b_raw = nn.Parameter(torch.tensor(0.0))          # sigmoid → (0, 2)
        # --- 설계 상수 (버퍼). 혼합 가중의 규모만 쓴다 — U·A 는 더 이상 쓰지 않는다.
        d = config.UNIT_DESIGN
        self.register_buffer("m_d", torch.tensor([d[u]["m_d"] for u in units]))

    # --- 물리 모수 (유계)
    @property
    def mu_jt(self) -> torch.Tensor:
        lo, hi = config.MU_JT_MIN, config.MU_JT_MAX
        return lo + (hi - lo) * torch.sigmoid(self.mu_jt_raw)

    @property
    def tg_b(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.tg_b_raw)

    # ------------------------------------------------------------------ 물리층
    def beta(self, valve):
        """바이패스 분율 — 밸브개도의 **단조 증가** 함수. 실측 β 와 Spearman 0.87/0.84."""
        h = self.beta_a + (nn.functional.softplus(self.beta_c)
                           * torch.relu(valve.unsqueeze(-1) - self.valve_knots)).sum(-1)
        return torch.sigmoid(h)

    def heater(self, eps, bath, t_in, valve):
        """(Q3)+(Q4). ε 은 신경망이 예측한 **관측량**이고 역산하지 않는다.

        가용성 `avail_u` — `ε_u` 는 구동온도차 `T_bath,u − T_in` 이 작으면 정의되지 않는다
        (분모가 0 에 가까워 잡음으로 발산). 그 히터를 혼합에서 **부드럽게 빼야** 한다.

        ⚠ 미래 시점의 가용성은 **관측할 수 없다** — `y_avail` 을 쓰면 누수다. 구동온도차는
          이미 예측하는 상태(`T_bath,u`·`T_in`)로 정해지므로 **거기서 계산**한다. 별도 헤드도,
          누수도 없다. 둘 다 요구하면 표본이 43,849시간 → 1,376행으로 붕괴한다(8차 버그).
        """
        t_in_e = t_in.unsqueeze(-1)
        drive = bath - t_in_e
        avail = torch.sigmoid((drive - config.MIN_DRIVE_C) / config.DRIVE_SOFT_C)
        t_out = t_in_e + eps * drive                            # (Q3)
        w = self.m_d * eps.clamp_min(1e-3) * avail              # 경험 가중 (문서 10)
        tot = w.sum(-1, keepdim=True)
        # 전부 비가용이면 혼합온도가 정의되지 않는다 → 균등 가중으로 되돌린다
        f = torch.where(tot > 1e-6, w / tot.clamp_min(1e-9),
                        torch.full_like(w, 1.0 / w.shape[-1]))
        b = self.beta(valve)
        t_hdr = (1.0 - b) * (f * t_out).sum(-1) + b * t_in      # (Q4)
        return t_hdr, t_out, f, b, avail

    def supply(self, t_hdr, dp, t_in, n_pipe):
        """(Q5)+(Q6). T_61 = T_g + (T_hdr − μ_JT·ΔP − T_g)·exp(−N)"""
        tg = self.tg_a + self.tg_b * t_in
        return tg + (t_hdr - self.mu_jt * dp - tg) * torch.exp(-n_pipe)

    # ------------------------------------------------------------------ 헤드
    def heads(self, x: torch.Tensor):
        h = self.trunk(x)
        o = self.state(h)
        mu, sd = o[:, 0::2], nn.functional.softplus(o[:, 1::2]) + 1e-2
        n_pipe = nn.functional.softplus(self.ntu_pipe(h)).squeeze(-1)
        return h, mu, sd, n_pipe

    def forward(self, x: torch.Tensor, n_mc: int = config.N_MC):
        """미래 예측. 상태 표본을 물리층에 통과시킨다."""
        h, mu, sd, n_pipe = self.heads(x)
        s = mu.unsqueeze(0) + sd.unsqueeze(0) * torch.randn(
            n_mc, *mu.shape, device=x.device)                   # (K, B, n_state)
        nu = len(self.units)
        t_in, dp, valve = s[..., 0], s[..., 1], s[..., 2]
        bath = s[..., 3:3 + nu]
        eps = s[..., 3 + nu:].clamp(config.EPS_LO, config.EPS_HI)
        t_hdr, t_out, f, beta, avail = self.heater(eps, bath, t_in, valve)
        t61 = self.supply(t_hdr, dp, t_in, n_pipe.unsqueeze(0))
        parts = {"mu": mu, "sd": sd, "n_pipe": n_pipe, "t_hdr": t_hdr, "t_out": t_out,
                 "f": f, "beta": beta, "eps": eps, "avail": avail, "t61": t61}
        return t61.mean(0), t61.std(0) + 1e-3, parts

    def now(self, x: torch.Tensor, eps, bath, t_in, dp, valve):
        """동시점 물리 정합용. 상태는 **실측**을 그대로 쓴다."""
        _, _, _, n_pipe = self.heads(x)
        t_hdr, t_out, f, beta, avail = self.heater(eps, bath, t_in, valve)
        t61 = self.supply(t_hdr, dp, t_in, n_pipe)
        return {"t_hdr": t_hdr, "t_out": t_out, "f": f, "beta": beta,
                "avail": avail, "t61": t61}

    def prob_freeze(self, x: torch.Tensor, n_mc: int = 256) -> torch.Tensor:
        _, _, parts = self.forward(x, n_mc)
        return (parts["t61"] < config.FREEZE_C).float().mean(0)


def nll(mu: torch.Tensor, sd: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (torch.log(sd) + 0.5 * ((y - mu) / sd) ** 2).mean()
