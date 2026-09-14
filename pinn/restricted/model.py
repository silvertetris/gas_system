"""제한 PINN — (Q5) 줄-톰슨 + (Q6) 매설배관만.

    T_61 = T_g + (T_hdr − μ_JT·ΔP − T_g)·exp(−N)
    N    = U·A_배관/(m·c_p) + N₀
    T_g  = a + b·T_in

신경망은 **관측 가능한 미래 상태만** 낸다: 헤더온도·압력강하·입구온도·log 유량.
물리층의 모수 μ_JT·U·A_배관·N₀·a·b 는 전역 스칼라다 — 해석 가능한 결과로 보고한다.

## 구조 인자 (Optuna 탐색 대상)

`hidden`·`n_layers`·`dropout`. 기본값(64·2·0)이면 **이전 판과 모듈 구성·초기화 순서가
완전히 같다** — 드롭아웃은 p>0 일 때만 모듈을 넣는다(p=0 모듈도 난수 소비 여부가 버전마다
달라 재현성을 해칠 수 있다).

## XAI 보조

  · `risk_score` — MC 표본 없이 **결정론적·미분 가능한** 위험점수 Φ((0 − T̂₆₁)/σ̂).
    T̂₆₁ 은 상태 평균을 물리층에 통과시킨 값, σ̂ 는 상태 표준편차의 **델타법 전파**.
    몬테카를로 P(빙결)는 지시함수라 기울기가 없어 SHAP 에 쓸 수 없다.
  · `decompose` — T_61 = T_g(1−e^{−N}) + e^{−N}·T_hdr − e^{−N}·μ_JT·ΔP 의 세 항.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from . import config

STATES = ("hdr", "dp", "t_in", "log_m")


class RestrictedPINN(nn.Module):
    def __init__(self, n_feat: int, ua_init_kw: float, hidden: int = config.HIDDEN,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers, d_in = [], n_feat
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        self.trunk = nn.Sequential(*layers)
        self.state = nn.Linear(hidden, 2 * len(STATES))
        lo, hi = config.MU_JT_MIN, config.MU_JT_MAX
        p0 = (config.MU_JT_INIT - lo) / (hi - lo)
        self.mu_jt_raw = nn.Parameter(torch.tensor(math.log(p0 / (1 - p0))))
        self.tg_a = nn.Parameter(torch.tensor(0.0))
        self.tg_b_raw = nn.Parameter(torch.tensor(0.0))                  # sigmoid → (0, 2)
        self.log_ua = nn.Parameter(torch.tensor(math.log(ua_init_kw * 1e3)))   # W/K
        self.n0_raw = nn.Parameter(torch.tensor(math.log(math.expm1(config.N0_INIT))))

    # --- 물리 모수
    @property
    def mu_jt(self):
        lo, hi = config.MU_JT_MIN, config.MU_JT_MAX
        return lo + (hi - lo) * torch.sigmoid(self.mu_jt_raw)

    @property
    def tg_b(self):
        return 2.0 * torch.sigmoid(self.tg_b_raw)

    @property
    def ua_kw(self):
        return torch.exp(self.log_ua) / 1e3

    @property
    def n0(self):
        return nn.functional.softplus(self.n0_raw)

    def n_pipe_log(self, log_m):
        """N = U·A/(m·c_p) + N₀ 를 **log m 으로** 계산한다: exp(log UA − log m)/c_p + N₀.

        ⚠ 2026-09-13 수정. 이전엔 표본 log m 을 exp 로 m 으로 바꾼 뒤 나눴다. 학습 초반 log m 헤드의
        표준편차가 18 까지 커지면 표본 log m 이 float32 한계(≈89)를 넘어 m = inf → 순전파는 N → N₀ 로
        멀쩡하지만 **기울기는 inf·0 = NaN** 이 되어 가중치가 망가졌다(M·AE 최종학습 시드 3/5 붕괴,
        Optuna trial 일부 CV AUC 0.49~0.58). 같은 식을 로그 공간에서 계산하면 넘침이 없다.
        """
        return torch.exp(self.log_ua - log_m.clamp_min(math.log(config.M_MIN))) / config.CP_GAS_J_KGK + self.n0

    def n_pipe(self, m):
        return self.n_pipe_log(torch.log(m.clamp_min(config.M_MIN)))

    def supply_log(self, hdr, dp, t_in, log_m):
        """(Q5)+(Q6), 유량을 log m 으로 받는다."""
        tg = self.tg_a + self.tg_b * t_in
        return tg + (hdr - self.mu_jt * dp - tg) * torch.exp(-self.n_pipe_log(log_m))

    def supply(self, hdr, dp, t_in, m):
        """(Q5)+(Q6)."""
        return self.supply_log(hdr, dp, t_in, torch.log(m.clamp_min(config.M_MIN)))

    def decompose(self, hdr, dp, t_in, log_m):
        """T_61 의 세 항: 지중 T_g(1−e) · 헤더 e·T_hdr · 줄-톰슨 −e·μ·ΔP  (합 = supply). 유량은 log m."""
        tg = self.tg_a + self.tg_b * t_in
        e = torch.exp(-self.n_pipe_log(log_m))
        return {"지중": tg * (1 - e), "헤더": e * hdr, "줄톰슨": -e * self.mu_jt * dp, "e": e}

    # --- 신경망
    def heads(self, x):
        o = self.state(self.trunk(x))
        return o[:, 0::2], nn.functional.softplus(o[:, 1::2]) + 1e-2

    def forward(self, x, n_mc: int = config.N_MC):
        mu, sd = self.heads(x)
        s = mu.unsqueeze(0) + sd.unsqueeze(0) * torch.randn(n_mc, *mu.shape, device=x.device)
        t61 = self.supply_log(s[..., 0], s[..., 1], s[..., 2], s[..., 3])
        return t61.mean(0), t61.std(0) + 1e-3, mu, sd, t61

    def prob_freeze(self, x, n_mc: int = 256):
        return (self.forward(x, n_mc)[4] < config.FREEZE_C).float().mean(0)

    # --- XAI 보조 (결정론적·미분 가능)
    def certainty_equivalent(self, x):
        mu, sd = self.heads(x)
        hdr, dp, tin, lm = mu.unbind(-1)
        n = self.n_pipe_log(lm)
        e = torch.exp(-n)
        tg = self.tg_a + self.tg_b * tin
        t61 = tg + (hdr - self.mu_jt * dp - tg) * e
        # 델타법: ∂f/∂[T_hdr, ΔP, T_in, log m]
        grad = torch.stack([e, -self.mu_jt * e, self.tg_b * (1 - e),
                            (hdr - self.mu_jt * dp - tg) * e * (n - self.n0)], dim=-1)
        sigma = ((grad * sd) ** 2).sum(-1).clamp_min(1e-6).sqrt()
        return t61, sigma

    def risk_score(self, x):
        t61, sigma = self.certainty_equivalent(x)
        z = (config.FREEZE_C - t61) / sigma
        return 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))


def nll(mu, sd, y, w=None):
    v = torch.log(sd) + 0.5 * ((y - mu) / sd) ** 2
    if w is None:
        return v.mean()
    return (v * w).sum() / w.sum().clamp_min(1.0)


class SoftPINN(RestrictedPINN):
    """물리 = **손실 제약**(soft PINN). 신경망이 공급온도 T61 을 직접 낸다 (2026-09-13, 문서 19).

    hard(RestrictedPINN)는 최종 출력이 반드시 (Q5)+(Q6) 를 거쳐, 식이 흐린 만큼 정확도가 막혔다 —
    미래 상태를 완벽히 알아도 물리식 T61 의 시험 AUC 가 M 0.79 · Z 0.76 으로 "지금 값 그대로" 규칙
    (0.87 · 0.86)보다 낮았다 (`audit/target_ceiling.py`). 여기서는
      · T61 헤드(평균·표준편차)가 타깃을 직접 맞추고,
      · 상태 헤드(헤더·ΔP·입구온도·log m)는 그대로 지도하며,
      · **정합 잔차** (T̂61 + d − f(상태̂))² / Var 로 두 출력을 물리식에 묶는다 (표준 PINN 잔차항),
      · 동시점 물리 잔차(관측 상태 → 관측 1시간 평균 T61)로 물리 모수를 데이터에 맞춘다 (hard 와 같음).
    d ≥ 0 은 분 단위 최저 타깃일 때만 쓰는 **순간 하강** 헤드 — 물리식은 1시간 평균을 기술하므로
    최저 = 평균 − d. 실제 1시간 평균(y_t61_mean)으로 T̂61 + d 를 지도한다.
    `use_physics=False` 면 물리 잔차·정합 잔차가 모두 빠진 **같은 신경망** (대조군 nn).
    빙결 확률은 해석적 Φ((0 − μ)/σ) — MC 가 없어 순위 동률이 없다.
    """

    def __init__(self, n_feat: int, ua_init_kw: float, hidden: int = config.HIDDEN, n_layers: int = 2,
                 dropout: float = 0.0, use_physics: bool = True, dip: bool = False):
        super().__init__(n_feat, ua_init_kw, hidden, n_layers, dropout)
        self.use_physics, self.dip = use_physics, dip
        self.t61_head = nn.Linear(hidden, 3 if dip else 2)

    def outputs(self, x):
        h = self.trunk(x)
        o, t = self.state(h), self.t61_head(h)
        smu, ssd = o[:, 0::2], nn.functional.softplus(o[:, 1::2]) + 1e-2
        mu, sd = t[:, 0], nn.functional.softplus(t[:, 1]) + 1e-2
        d = nn.functional.softplus(t[:, 2]) if self.dip else torch.zeros_like(mu)
        return mu, sd, smu, ssd, d

    def forward(self, x, n_mc: int = config.N_MC):
        mu, sd, smu, ssd, _ = self.outputs(x)
        return mu, sd, smu, ssd, mu.unsqueeze(0) + sd.unsqueeze(0) * torch.randn(n_mc, *mu.shape, device=x.device)

    def certainty_equivalent(self, x):
        mu, sd = self.outputs(x)[:2]
        return mu, sd

    def prob_freeze(self, x, n_mc: int = 256):
        return self.risk_score(x)


class _LSTMTrunk(nn.Module):
    """(배치, 창, 특징) → 마지막 시각의 은닉 (배치, hidden)."""

    def __init__(self, n_in: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.drop(out[:, -1])


class LSTMNet(SoftPINN):
    """시계열 LSTM 대조군 (2026-09-14, 문서 21). nn 과 **헤드·손실이 같고 몸통만 LSTM** 이다.

    입력은 시각 t 까지 과거 `window` 시간의 표준화 특징 + 행 존재 마스크 (배치, 창, 특징+1).
    출력은 T61 직접(평균·σ) + 6h 뒤 상태 — nn 과 동일, 물리 잔차 없음. 빙결 확률 Φ((0−μ)/σ).
    """

    def __init__(self, n_feat: int, ua_init_kw: float, hidden: int = config.HIDDEN, n_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__(n_feat, ua_init_kw, hidden, 1, 0.0, use_physics=False, dip=False)
        self.trunk = _LSTMTrunk(n_feat, hidden, n_layers, dropout)


def make(arch: str, n_feat: int, ua_init_kw: float, hidden: int, n_layers: int, dropout: float,
         dip: bool = False) -> RestrictedPINN:
    """hard = RestrictedPINN (초기화 순서 기존과 동일) · soft = SoftPINN · nn = SoftPINN(물리 없음) · lstm = LSTMNet."""
    if arch == "lstm":
        return LSTMNet(n_feat, ua_init_kw, hidden, n_layers, dropout)
    if arch == "hard":
        return RestrictedPINN(n_feat, ua_init_kw, hidden, n_layers, dropout)
    if arch in ("soft", "softw", "nn"):
        return SoftPINN(n_feat, ua_init_kw, hidden, n_layers, dropout, use_physics=arch != "nn",
                        dip=dip and arch != "nn")
    raise ValueError(f"알 수 없는 구조: {arch}")
