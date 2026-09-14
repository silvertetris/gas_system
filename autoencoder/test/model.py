"""LSTM Autoencoder — 시퀀스 → 잠재벡터 → 시퀀스 복원.

구조: Encoder LSTM 의 마지막 은닉상태를 latent_dim 으로 사영 → 그 벡터를 L 번 반복해
Decoder LSTM 에 넣고 원 시퀀스를 복원한다(repeat-vector 방식).

**왜 이 구조인가**: 잠재벡터가 윈도우 전체를 요약한 고정길이 상태가 되므로
PINN 의 상태변수 후보로 바로 넘길 수 있다. 시점별 잠재표현이 필요하면
seq2seq(어텐션) 로 바꿔야 하지만, 정제 목적에는 이 정도가 맞다.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from lstm.test.model import get_device, set_seed  # noqa: F401  (동일 유틸 재사용)

from . import config

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, latent_dim: int,
                 hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        drop = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers,
                               batch_first=True, dropout=drop)
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True, dropout=drop)
        self.out = nn.Linear(hidden_size, n_features)
        self.drop = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.to_latent(self.drop(h[:, -1, :]))      # (N, latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h0 = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        h, _ = self.decoder(h0)
        return self.out(h)                                  # (N, L, F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def _loader(X: np.ndarray, batch_size: int, shuffle: bool):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(X_tr: np.ndarray, X_val: np.ndarray, params: dict, epochs: int,
                patience: int | None = None, device: torch.device | None = None,
                verbose: bool = False) -> tuple[LSTMAutoencoder, float]:
    """재구성 MSE 로 학습. 반환: (모델, 최적 검증 RMSE[스케일 단위])."""
    device = device or get_device()
    model = LSTMAutoencoder(X_tr.shape[-1], X_tr.shape[1], params["latent_dim"],
                            params["hidden_size"], params["num_layers"],
                            params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])
    lossf = nn.MSELoss()
    tr_dl = _loader(X_tr, params["batch_size"], True)
    va_dl = _loader(X_val, params["batch_size"], False)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for (xb,) in tr_dl:
            xb = xb.to(device)
            opt.zero_grad()
            lossf(model(xb), xb).backward()
            opt.step()

        model.eval()
        se = cnt = 0.0
        with torch.no_grad():
            for (xb,) in va_dl:
                xb = xb.to(device)
                se += float(((model(xb) - xb) ** 2).sum())
                cnt += xb.numel()
        rmse = (se / max(cnt, 1)) ** 0.5
        if verbose:
            logger.info("  epoch %2d  val recon RMSE(scaled) %.5f", ep + 1, rmse)
        if rmse < best - 1e-6:
            best, bad = rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience and bad >= patience:
                if verbose:
                    logger.info("  early stop @ epoch %d", ep + 1)
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


def reconstruct(model: LSTMAutoencoder, X: np.ndarray, batch_size: int = 256,
                device: torch.device | None = None) -> np.ndarray:
    device = device or get_device()
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in _loader(X, batch_size, False):
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def encode_all(model: LSTMAutoencoder, X: np.ndarray, batch_size: int = 256,
               device: torch.device | None = None) -> np.ndarray:
    """잠재벡터 추출 — PINN 상태표현 후보로 넘길 산출물."""
    device = device or get_device()
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in _loader(X, batch_size, False):
            out.append(model.encode(xb.to(device)).cpu().numpy())
    return np.concatenate(out)
