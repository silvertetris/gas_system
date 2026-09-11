"""LSTM 회귀 모델 + 학습/평가 루프 (PyTorch)."""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from . import config

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = config.SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LSTMRegressor(nn.Module):
    """다변수 시퀀스 → 스칼라 회귀. 마지막 타임스텝 은닉상태만 쓴다."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,   # 1층이면 torch가 무시하고 경고만 낸다
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_model(X_tr, y_tr, X_val, y_val, params: dict, epochs: int,
                patience: int | None = None, device: torch.device | None = None,
                verbose: bool = False) -> tuple[LSTMRegressor, float]:
    """학습 후 (모델, 최종 검증 RMSE[스케일된 단위]) 반환.

    ⚠ shuffle=True 는 **윈도우 단위**로만 섞는다. 윈도우 내부 시간순서는 유지되고
      학습/검증 분할도 이미 시간순이라 시계열 원칙을 깨지 않는다.
    """
    device = device or get_device()
    model = LSTMRegressor(X_tr.shape[-1], params["hidden_size"],
                          params["num_layers"], params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])
    lossf = nn.MSELoss()
    tr_dl = _loader(X_tr, y_tr, params["batch_size"], shuffle=True)
    va_dl = _loader(X_val, y_val, params["batch_size"], shuffle=False)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()

        model.eval()
        se = cnt = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                se += float(((model(xb) - yb) ** 2).sum())
                cnt += len(yb)
        rmse = (se / max(cnt, 1)) ** 0.5
        if verbose:
            logger.info("  epoch %2d  val RMSE(scaled) %.4f", ep + 1, rmse)
        if rmse < best - 1e-5:
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


def predict(model: LSTMRegressor, X: np.ndarray, batch_size: int = 256,
            device: torch.device | None = None) -> np.ndarray:
    device = device or get_device()
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(X)), batch_size=batch_size):
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)
