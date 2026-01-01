from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .logit import MulticlassLogit


# -------------------------------------------------------------------------- #
#  TreatmentConfig
# -------------------------------------------------------------------------- #
@dataclass
class TreatmentConfig:
    r"""
    Parameters
    ----------
    n_treatments : int
        総処置カテゴリ数 :math:`K`.
    beta_x : ndarray[float]
    beta_w : ndarray[float]
    seed : int | None, default=None
        乱数シード（`numpy.random.default_rng` に渡す）。
    """

    n_treatments: int
    beta_x: NDArray[np.floating]
    # beta_w: NDArray[np.floating]
    seed: int | None = None


# -------------------------------------------------------------------------- #
#  TreatmentModel
# -------------------------------------------------------------------------- #
class TreatmentModel:
    r"""
    多項ロジスティック回帰で処置 :math:`T` を生成するモデル

    $$
        \Pr(T=t \mid X,W)
        \;=\;
        \frac{\exp(X\beta^X_t + W\beta^W_t)}
             {1 + \sum_{k=1}^{K-1} \exp(X\beta^X_k + W\beta^W_k)},
        \qquad t=0,\dots,K-1,
    $$
    ただし基準カテゴリ :math:`t=0` のロジットは 0 に固定。
    """

    # -------------------- construction ---------------------------------- #
    def __init__(self, cfg: TreatmentConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        K = cfg.n_treatments
        self._beta_x = np.asarray(cfg.beta_x, dtype=float)  # (K, p)
        # self._beta_w = np.asarray(cfg.beta_w, dtype=float)  # (K, q)

        if self._beta_x.shape[0] != K:
            raise ValueError("beta_x の行数は K でなければなりません")
        # if self._beta_w.shape[0] != K:
        #     raise ValueError("beta_w の行数は K でなければなりません")

        self.model = MulticlassLogit(self._beta_x, self.rng)

    def __call__(self, X: NDArray) -> NDArray[np.int64]:
        """
        処置割当を返す。OutcomeModel と同じく ``__call__`` で生成。

        Returns
        -------
        T : (n,) ndarray[int]
        """
        return self.model.sample(X)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # データ
    n, p, q = 100, 3, 2
    X = rng.standard_normal((n, p))
    W = rng.standard_normal((n, q))

    # 係数（K=3 ⇒ 行数=2）
    beta_x = np.array([[2.0, 0.0, 0.0], [0.0, 2.5, 0.0]])
    beta_w = np.array([[-1.5, 0.0], [0.0, -2.0]])

    cfg = TreatmentConfig(
        n_treatments=3,
        beta_x=beta_x,
        beta_w=beta_w,
        seed=42,
    )
    treat_model = TreatmentModel(cfg)

    T = treat_model(X, W)
    print("処置の出現頻度:", np.bincount(T))
