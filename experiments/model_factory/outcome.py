from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit


@dataclass
class OutcomeConfig:
    r"""
    Configuration container for :class:`OutcomeDataFactory`.

    Parameters
    ----------
    common_coeffs : (p,) ndarray[float]
        Vector :math:`\beta` – common linear effects of the :math:`X` covariates.
    effects : dict[tuple[int, int], float]
        Dictionary mapping a pair ``(treatment, version) -> tau_{t,v}``.
    desired_snr : float, default=5.0
        Desired signal–noise ratio

        .. math::
            \mathrm{SNR} = \frac{\sigma_{\text{signal}}}{\sigma_{\text{noise}}}

    binomial : bool, default=False
        If ``True``, apply logistic link and return Bernoulli draws.
    seed : int | None, default=None
        Seed for the internal random generator.
    """

    common_coeffs: dict[tuple[int, int], NDArray]
    effects: dict[tuple[int, int], NDArray]
    desired_snr: float = 5.0
    binomial: bool = False
    seed: int | None = None


class OutcomeModel:
    r"""
    アウトカムモデルに基づきデータを生成する
    """

    def __init__(self, config: OutcomeConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def _noise(self, signal: NDArray[np.floating]) -> NDArray[np.floating]:
        r"""
        Draw Gaussian noise whose standard deviation satisfies

        \[
          \sigma_\text{noise}
          =
          \frac{\sigma_\text{signal}}{\text{SNR}}\,
          \times\,
          \text{noise_scale}.
        \]
        """
        sd_signal = float(signal.std(ddof=0))
        sd_noise = sd_signal / self.cfg.desired_snr
        return self.rng.normal(loc=0.0, scale=sd_noise, size=signal.shape[0])

    def __call__(
        self,
        X: NDArray[np.floating],
        treatment: NDArray[np.integer],
        version: NDArray[np.integer],
    ) -> NDArray[np.floating]:
        r"""
        Parameters
        ----------
        X : (n, p) ndarray[float]
            Covariate matrix.
        treatment : (n,) ndarray[int]
            Treatment assignment.
        version : (n,) ndarray[int]
            Version assignment.

        Returns
        -------
        Y : (n,) ndarray[float] | ndarray[int]
            Continuous or Bernoulli outcomes, depending on
            ``config.binomial_outcome``.
        """
        n, p = X.shape

        # -------------------------------------------------- #
        # 1) Linear part  X β ... common effects
        # -------------------------------------------------- #
        y_lin = np.zeros(n, dtype=np.float64)
        for (t, v), beta_tv in self.cfg.common_coeffs.items():
            mask = (treatment == t) & (version == v)
            y_lin[mask] = X[mask] @ beta_tv  # shape (n,)

        # -------------------------------------------------- #
        # 2) Treatment-version effects \tau_{t,v}
        # -------------------------------------------------- #

        tau = np.zeros(n, dtype=np.float64)
        for (t, v), eff in self.cfg.effects.items():
            mask = (treatment == t) & (version == v)
            tau[mask] = eff

        # -------------------------------------------------- #
        # 3) Additive noise
        # -------------------------------------------------- #
        noise = self._noise(y_lin + tau)

        y = y_lin + tau + noise

        # -------------------------------------------------- #
        # 4) Optional logistic transform
        # -------------------------------------------------- #
        if self.cfg.binomial:
            probs = expit(y)
            y = self.rng.binomial(1, probs)

        return y


if __name__ == "__main__":
    # ―――― 構成の宣言 ――――
    cfg = OutcomeConfig(
        common_coeffs=np.ones(10),
        effects={(0, 0): 0.5, (1, 0): 1.2, (1, 1): 2.0},
        desired_snr=5,
        binomial=False,
        seed=42,
    )

    # ―――― ファクトリを初期化 ――――
    outcome_factory = OutcomeModel(cfg)

    # ―――― ダミーの入力データ ――――
    n = 10
    X = np.random.randn(n, 10)
    T = np.random.randint(0, 2, size=n)  # 2 treatments
    V = np.random.randint(0, 2, size=n)  # 2 versions per treatment

    # ―――― アウトカム生成 ――――
    Y = outcome_factory(X, T, V)

    print(Y)
