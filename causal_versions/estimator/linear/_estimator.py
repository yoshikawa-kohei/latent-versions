from typing import Optional, Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_consistent_length


class LinearModel(BaseEstimator, RegressorMixin):
    """
    線形回帰モデルのパラメータを推定するクラス


    Parameters
    ----------
    fit_intercept : bool, default=True
        切片を含めるかどうか
    """

    def __init__(self, fit_intercept: bool = True, **kwargs) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.sigma2_: Optional[float] = None
        self.sample_weight: Optional[npt.ArrayLike] = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """特徴量行列に切片用の列を追加"""
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate([intercept, X], axis=1)
        return X

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, sample_weight: Optional[npt.ArrayLike] = None, **kwargs) -> Self:
        """
        重み付きガウシアンモデルのパラメータを推定する

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            特徴量行列
        y : array-like, shape (n_samples,)
            目的変数
        sample_weight : array-like, shape (n_samples,), optional
            各サンプルの重み
            Noneの場合は全てのサンプルを均等に扱う

        Returns
        -------
        self : object
            パラメータが推定されたモデル
        """
        # 入力チェック
        X_array: np.ndarray = check_array(X, ensure_2d=True, dtype=np.float64)
        y_array: np.ndarray = check_array(y, ensure_min_samples=1, ensure_2d=False, dtype=np.float64)
        check_consistent_length(X_array, y_array)

        # 切片用の列を追加
        X_with_intercept: np.ndarray = self._add_intercept(X_array)

        # 重み行列　W を作成
        if sample_weight is None:
            # sample_weight が None の場合は、対角行列として扱う（つまり重みつきにしない）
            W: np.ndarray = np.eye(X_array.shape[0])
            weights_array: np.ndarray = np.ones(X_array.shape[0])
        else:
            weights_array = check_array(sample_weight, ensure_min_samples=1, ensure_2d=False, dtype=np.float64)
            check_consistent_length(X_array, weights_array)
            self.sample_weight = weights_array
            # 重み行列（対角行列）
            W = np.diag(weights_array)

        # 平均パラメータ coef_ の更新
        # coef_ = (X^T P X)^{-1} X^T P y
        XtP: np.ndarray = X_with_intercept.T @ W
        XtPX: np.ndarray = XtP @ X_with_intercept
        XtPy: np.ndarray = XtP @ y_array
        try:
            self.coef_ = np.linalg.solve(XtPX, XtPy)
        except np.linalg.LinAlgError as e:
            print(e)
            sw = np.sqrt(np.clip(weights_array, 1e-12, None)).reshape(-1, 1)
            Zs = X_with_intercept * sw
            ys = y_array * sw
            self.coef_, *_ = np.linalg.lstsq(Zs, ys, rcond=1e-10)

        # 予測値の計算
        y_pred: np.ndarray = X_with_intercept @ self.coef_

        # 分散パラメータ sigma^2 の更新
        # sigma^2 = (y - X coef_)^T P (y - X coef_) / (1^T P 1)
        residuals: np.ndarray = y_array - y_pred
        numerator: float = float(residuals.T @ W @ residuals)
        denominator: float = float(np.sum(weights_array))  # 1^T P 1
        self.sigma2_ = numerator / denominator

        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """
        予測を行う

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            予測したいデータの特徴量

        Returns
        -------
        array-like, shape (n_samples,)
            予測値
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() before predict().")

        X_array: np.ndarray = check_array(X, ensure_2d=True, dtype=np.float64)
        X_with_intercept: np.ndarray = self._add_intercept(X_array)
        return X_with_intercept @ self.coef_

    def score(self, X: npt.ArrayLike, y: npt.ArrayLike, sample_weight: Optional[npt.ArrayLike] = None) -> float:
        """
        モデルの対数尤度を計算する

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            テストデータの特徴量
        y : array-like, shape (n_samples,)
            テストデータの目的変数
        sample_weight : array-like, shape (n_samples,), optional
            各サンプルの重み（デフォルトはNone、その場合は均等な重みを使用）

        Returns
        -------
        float
            重み付き対数尤度
        """
        if self.coef_ is None or self.sigma2_ is None:
            raise ValueError("モデルが学習されていません。先にfit()を呼び出してください。")

        X_array: np.ndarray = check_array(X, ensure_2d=True, dtype=np.float64)
        y_array: np.ndarray = check_array(y, ensure_min_samples=1, ensure_2d=False, dtype=np.float64)
        check_consistent_length(X_array, y_array)

        # 予測値を計算
        y_pred: np.ndarray = self.predict(X_array)

        # 対数尤度を計算
        log_likelihood: np.ndarray = -0.5 * np.log(2 * np.pi * self.sigma2_) - (1 / (2 * self.sigma2_)) * (y_array - y_pred) ** 2

        # 重み付き対数尤度の合計を返す
        if sample_weight is not None:
            weights_array: np.ndarray = check_array(sample_weight, ensure_min_samples=1, ensure_2d=False, dtype=np.float64)
            check_consistent_length(X_array, weights_array)
            return float(np.sum(weights_array * log_likelihood))
        else:
            return float(np.sum(log_likelihood))

    def predict_proba(self, X: npt.ArrayLike, y: npt.ArrayLike, sample_weight: Optional[npt.ArrayLike] = None) -> float:
        """
        モデルの確率を計算する

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            特徴量
        y : array-like, shape (n_samples,)
            目的変数
        sample_weight : array-like, shape (n_samples,), optional
            各サンプルの重み（デフォルトはNone、その場合は均等な重みを使用）

        Returns
        -------
        float
            重み付き対数尤度
        """

        y_pred = self.predict(X)
        # 正規分布の確率密度関数

        return np.exp(-((y - y_pred) ** 2) / (2 * self.sigma2_)) / np.sqrt(2 * np.pi * self.sigma2_)

    def conditional_density(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: Optional[npt.ArrayLike] = None,
    ) -> npt.NDArray[np.float64]:
        return self.predict_proba(X, y, sample_weight)

    def log_likelihood(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: Optional[npt.ArrayLike] = None,
    ) -> float:
        """
        モデルの対数尤度を計算する

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            特徴量
        y : array-like, shape (n_samples,)
            目的変数

        Returns
        -------
        float
            重み付き対数尤度
        """

        y_pred = self.predict(X)
        # 正規分布の確率密度関数
        log_likelihood = -0.5 + np.log(2 * np.pi * self.sigma2_) - ((y - y_pred) ** 2) / (2 * self.sigma2_)

        if sample_weight is not None:
            return np.sum(sample_weight * log_likelihood)
        else:
            return np.sum(log_likelihood)

    def set_kernel_params(self, **kwargs) -> None:
        """
        カーネルパラメータを設定するメソッド
        現在は何も行わないが、将来の拡張のために残しておく
        """
        pass

    def log_conditional_density(self, X, y):
        y_pred = self.predict(X)

        log_density = -0.5 * np.log(2 * np.pi * self.sigma2_) - ((y - y_pred) ** 2) / (2 * self.sigma2_)

        return log_density
        # if self.sample_weight is not None:
        #     return np.sum(self.sample_weight * log_density)
        # else:
