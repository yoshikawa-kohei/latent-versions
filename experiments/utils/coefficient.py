from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .generator import GeneratorConfig


class CoefficientFactory:
    """
    GeneratorConfig に従って β を自動生成する補助クラス
    """

    def __init__(self, cfg: "GeneratorConfig", rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    # ---- Treatment 用 β ---- #
    def treatment_coeff(self, p: int) -> tuple[NDArray, NDArray]:
        K = self.cfg.n_treatments
        beta_x = np.zeros((K, p))
        lam = self.cfg.treatment_strength

        for t in range(1, K):
            j1 = (2 * t - 2) % p
            j2 = (2 * t - 1) % p
            beta_x[t, j1] = +lam
            beta_x[t, j2] = -lam

        return beta_x

    # ---- Version 用 β ---- #
    def version_coeff(self, p: int) -> tuple[list[NDArray], list[NDArray]]:
        bx_list = []
        lam = self.cfg.version_strength
        K = self.cfg.n_treatments

        offset = 2 * (K - 1)
        for t, K_t in enumerate(self.cfg.n_versions):
            if K_t <= 0:
                bx_list.append(np.empty((0, p)))
                continue

            bx = np.zeros((K_t, p))
            for v in range(1, K_t):
                j1 = (2 * v - 2) % p + offset
                j2 = (2 * v - 1) % p + offset
                bx[v, j1] = +lam
                bx[v, j2] = +lam
            bx_list.append(bx)

        return bx_list

    # ---- Outcome 用パラメータ ---- #
    def outcome_params(self) -> tuple[NDArray, dict[tuple[int, int], float]]:
        p = self.cfg.covariate_dim
        beta_common = np.ones(p) * self.cfg.common_effect
        beta_common /= np.linalg.norm(beta_common)

        # version-specific perturbations (識別性のためのずらし)
        beta: dict[tuple[int, int], NDArray] = {}
        effects: dict[tuple[int, int], float] = {}

        eps = 0.2  # ずらしの強さ（小さめ）
        basis = np.eye(p)
        eff = 1.0

        effects: dict[tuple[int, int], float] = {}
        eff = 1.0
        for t, V_t in enumerate(self.cfg.n_versions):
            for v in range(V_t):
                beta[(t, v)] = beta_common + eps * basis[v]  # 方向を変えるだけ
                effects[(t, v)] = eff
                eff += 1.0
        return beta, effects
