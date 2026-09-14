"""PCA 베이스라인 — 선형 AE 와 등가.

AE 가 PCA 를 못 이기면 **비선형을 쓸 이유가 없다**. LSTM 실험에서 persistence 가 했던 역할.
윈도우 (N, L, F) 를 (N, L*F) 로 펼쳐 PCA 를 적용한다.
⚠ PCA 도 **학습 구간으로만 fit** 한다.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def pca_reconstruction_rmse(X_tr: np.ndarray, X_te: np.ndarray, n_components: int
                            ) -> tuple[float, np.ndarray]:
    """Returns: (테스트 재구성 RMSE[원 단위], 재구성값 (N,L,F))."""
    n, L, F = X_te.shape
    flat_tr = X_tr.reshape(len(X_tr), -1)
    flat_te = X_te.reshape(n, -1)
    pca = PCA(n_components=n_components, random_state=0).fit(flat_tr)
    rec = pca.inverse_transform(pca.transform(flat_te))
    rmse = float(np.sqrt(np.mean((rec - flat_te) ** 2)))
    return rmse, rec.reshape(n, L, F)
