from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.model_factory import OutcomeConfig, OutcomeModel, TreatmentConfig, TreatmentModel, VersionConfig, VersionModel
from experiments.utils import CoefficientFactory


@dataclass(frozen=True)
class GeneratorConfig:
    n_samples: int = 1_000
    n_treatments: int = 3
    n_versions: tuple[int, ...] = (2, 2, 2)
    covariate_dim: int = 10  # (p, q)
    treatment_strength: float = 2.0
    version_strength: float = 2.0
    common_effect: float = 1.0
    outcome_snr: float = 10.0
    binomial: bool = False
    seed: int = 42


class SyntheticDataGenerator:
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self.fix_rng = np.random.default_rng(0)
        self.rng = np.random.default_rng(cfg.seed)

        # ---- 係数生成 ---- #
        coef_factory = CoefficientFactory(cfg, self.rng)
        p = cfg.covariate_dim

        # -- Treatment model --
        beta_x_t = coef_factory.treatment_coeff(p)
        t_cfg = TreatmentConfig(cfg.n_treatments, beta_x_t, cfg.seed)
        self.treatment_model = TreatmentModel(t_cfg)

        # -- Version model --
        beta_x_v = coef_factory.version_coeff(p)
        v_cfg = VersionConfig(list(cfg.n_versions), beta_x_v, cfg.seed)
        self.version_model = VersionModel(v_cfg)

        # -- Outcome model --
        self.common_effects, self.effects = coef_factory.outcome_params()
        o_cfg = OutcomeConfig(self.common_effects, self.effects, desired_snr=cfg.outcome_snr, binomial=cfg.binomial, seed=cfg.seed)
        self.outcome_model = OutcomeModel(o_cfg)

    # ------------------------------------------------------------------ #
    def _sample_covariates(self) -> tuple[NDArray, NDArray]:
        n, p = self.cfg.n_samples, self.cfg.covariate_dim
        X = self.rng.standard_normal((n, p))

        return X

    # ------------------------------------------------------------------ #
    def generate(self) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Returns
        -------
        X, W, T, V, Y : ndarray
        """
        X = self._sample_covariates()
        T = self.treatment_model(X)
        V = self.version_model(T, X)
        Y = self.outcome_model(X, T, V)
        return X, T, V, Y


# ====================================================================== #
# 5) ――― 動作確認 ―――
# ====================================================================== #
if __name__ == "__main__":
    cfg = GeneratorConfig(
        n_samples=500,
        n_treatments=3,
        n_versions=(2, 3, 1),
        covariate_dim=(3, 2),
        treatment_strength=2.0,
        version_strength=1.5,
        outcome_snr=5.0,
        binomial=False,
        seed=42,
    )

    gen = SyntheticDataGenerator(cfg)
    X, W, T, V, Y = gen.generate()

    print("X shape:", X.shape)
    print("T counts:", np.bincount(T))
    for t in range(cfg.n_treatments):
        print(f"   t={t}  V counts:", np.bincount(V[T == t]))
    print("Y mean / std:", Y.mean(), Y.std(ddof=0))
