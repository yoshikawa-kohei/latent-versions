from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax


class MulticlassLogit:
    """基底となる多項ロジットモデル（基準カテゴリのロジット=0）。"""

    def __init__(self, beta_x: NDArray, rng: np.random.Generator):
        self.bx = beta_x  # (C,p)
        self.C = beta_x.shape[0]
        self.rng = rng

    def logits(self, X: NDArray) -> NDArray:
        logit = X @ self.bx.T
        # zero_vec = np.zeros((X.shape[0], 1))

        return logit

    def sample(self, X: NDArray) -> NDArray[np.int64]:
        P = softmax(self.logits(X), axis=1)
        return np.array([self.rng.choice(self.C, p=p) for p in P], dtype=np.int64)
