from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .logit import MulticlassLogit


# -------------------------------------------------------------------------- #
#  VersionConfig
# -------------------------------------------------------------------------- #
@dataclass
class VersionConfig:
    r"""
    Parameters
    ----------
    n_versions_per_treatment : list[int]
        各処置 $t$ に存在するバージョン数
        ``[V₀, V₁, …, V_{K-1}]`` (`V_t ≥ 1`).
    beta_x : list[ndarray]
        ``beta_x[t]`` は形状 ``(V_t-1, p)``
        （基準バージョン *v = 0* を除く *v = 1,…,V_t-1* の係数行列）。
    seed : int | None, default=None
        `numpy.random.default_rng` へ渡すシード。
    """

    n_versions_per_treatment: list[int]
    beta_x: list[NDArray[np.floating]]
    seed: int | None = None


class VersionModel:
    def __init__(self, cfg: VersionConfig):
        self.models: list[MulticlassLogit] = []  # t ごとに MulticlassLogit を保持
        self.rng = np.random.default_rng(cfg.seed)
        for t, V_t in enumerate(cfg.n_versions_per_treatment):
            if V_t == 1:
                self.models.append(None)  # dummy
            else:
                self.models.append(
                    MulticlassLogit(cfg.beta_x[t], self.rng)  # 同一 RNG から派生
                )

    def __call__(self, T, X):
        V = np.zeros_like(T)
        for t in np.unique(T):
            idx = np.where(T == t)[0]
            if (model := self.models[t]) is not None:
                model: MulticlassLogit
                V[idx] = model.sample(X[idx])
        return V


if __name__ == "__main__":
    from experiments.model_factory import TreatmentConfig, TreatmentModel

    rng = np.random.default_rng(0)

    # ---- データ生成用共通変数 ----
    n, p, q = 300, 3, 2
    X = rng.standard_normal((n, p))
    W = rng.standard_normal((n, q))

    # ---- 1) TreatmentModel ----
    beta_x_t = np.array([[2.0, 0.0, 0.0], [0.0, 2.5, 0.0]])
    beta_w_t = np.array([[-1.5, 0.0], [0.0, -2.0]])

    treat_cfg = TreatmentConfig(
        n_treatments=3,
        beta_x=beta_x_t,
        beta_w=beta_w_t,
        seed=1,
    )
    T_model = TreatmentModel(treat_cfg)
    T = T_model(X, W)

    # ---- 2) VersionModel ----
    #   V₀=2, V₁=3, V₂=1
    beta_x_v = [
        np.array([[1.0, 0.0, 0.0]]),  # t=0, shape (1, p)
        np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),  # t=1, shape (2, p)
        np.empty((0, p)),  # t=2, V₂=1 → (0, p)
    ]
    beta_w_v = [
        np.zeros((1, q)),  # t=0
        np.zeros((2, q)),  # t=1
        np.empty((0, q)),  # t=2
    ]
    ver_cfg = VersionConfig(
        n_versions_per_treatment=[2, 3, 1],
        beta_x=beta_x_v,
        beta_w=beta_w_v,
        seed=2,
    )
    V_model = VersionModel(ver_cfg)
    V = V_model(T, X, W)

    print("処置頻度:", np.bincount(T))
    for t in range(3):
        mask = T == t
        print(f"t={t} のバージョン頻度:", np.bincount(V[mask]))
