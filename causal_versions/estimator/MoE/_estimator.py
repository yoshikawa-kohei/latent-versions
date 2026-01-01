from typing import Any

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from scipy.special import logsumexp

from causal_versions.estimator.linear import LinearModel
from causal_versions.estimator.mnlogit._estimator import MultinomialLogisticRegression


class MoE:
    """
    Mixture-of-experts via EM algorithm
    """

    def __init__(
        self,
        n_components=2,
        init_params="kmeans",
        max_iter=100,
        tol=1e-5,
        random_state=None,
    ):
        # ハイパーパラメータの定義
        self.n_components = n_components
        self.init_params = init_params
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # EM で求まるパラメータを格納する属性（fit で設定）
        # initialization :
        self.gate = MultinomialLogisticRegression()  # クラスタごとの混合係数関数 π_v (X_i, W_i), mixture of experts の gate 関数
        self.density: dict[Any, LinearModel] = {}  # 各クラスタのセミパラメトリック密度 f_v(Y_i | X_i) を保持する辞書
        for v in range(self.n_components):
            self.density[v] = LinearModel()
        self.posterior: np.ndarray

        self.log_likelihood_ = []  # 各イテレーションの対数尤度履歴
        self.n_iter_ = 0  # 実際に行ったイテレーション回数
        self.converged_ = False  # 収束フラグ

    def fit(self, Y, X):
        """
        X: 形状 (n_samples, n_features) の入力データ行列
        y: 使わない（混合モデルのため）ので None
        """
        # 特徴量名の保存
        if getattr(X, "columns", None) is not None:
            self.x_feature_names_in_ = X.columns.tolist()

        if getattr(Y, "name", None) is not None:
            self.y_feature_names_in_ = Y.name

        # データ検証と形状取得
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)
        self.n_samples, self.n_features_x = X.shape

        # 初期化処理
        # random initialization or k-means
        # k_means_model = KMeans(
        #     n_clusters=self.n_components,
        #     n_init=20,  # 複数初期化でより安定
        #     random_state=self.random_state,
        # )
        # gmm = GaussianMixture(n_components=self.n_components, init_params="k-means++", random_state=self.random_state, tol=1e-5)

        # init_data = np.concatenate([X], axis=1)
        # init_data = StandardScaler().fit_transform(init_data)

        # gmm.fit(init_data)
        rng = np.random.default_rng(self.random_state)
        alpha = np.ones(self.n_components)
        # shape: (n_samples, K)
        self.posterior = rng.dirichlet(alpha, size=self.n_samples)

        # self.posterior = np.zeros(
        #     (self.n_samples, self.n_components)
        # )  # 形: (n_samples, n_components)  # 各サンプル i に対する各クラスタ v の posterior 確率 p_iv (n_samples, n_components)

        # labels 番号を用いて、各クラスタの初期パラメータを設定
        # self.posterior[np.arange(self.n_samples), labels] = 1.0  # 初期責任度を設定

        # 対数尤度初期化
        prev_log_likelihood = -np.inf

        # Rich の Progress をカスタマイズ
        progress = Progress(
            TextColumn("[bold blue]EM Iteration[/]"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total} iter"),
            TimeElapsedColumn(),
            TextColumn(" | log_likelihood = "),
            TextColumn("[green]{task.fields[ll]:.6f}[/]"),
            ## ここで更新頻度を抑制
            refresh_per_second=1,  # デフォルトは 10 回／秒。1 にすれば１秒に１回更新
            transient=False,  # 最後の描画を残すかどうか
            redirect_stdout=False,  # stdout のリダイレクトを抑制
            redirect_stderr=False,
            disable=True,
        )
        with progress:
            task = progress.add_task("", total=self.max_iter, ll=prev_log_likelihood)

            for iteration in range(self.max_iter):
                # print(f"EM Iteration {iteration + 1}/{self.max_iter}")
                self.n_iter_ = iteration + 1

                # =============================
                # 1) M_step: パラメータ更新
                # =============================
                #   (a) 混合係数 π_v (X_i, W_i) を更新
                #   (b) アウトカムモデルの推定
                self._m_step(Y, X)

                # =============================
                # 2) E_step: 潜在変数期待値の計算
                # =============================
                # 各サンプル i, 各クラスタ v に対する「責任度」（posterior）を計算
                self._e_step(Y, X)

                # =============================
                # 3) 収束判定: 対数尤度を計算し、収束条件をチェック
                # =============================
                curr_log_likelihood = self.compute_log_likelihood(Y, X)
                self.log_likelihood_.append(curr_log_likelihood)

                # フィールドを更新して再描画
                progress.update(task, advance=1, ll=curr_log_likelihood)

                # 収束判定関数を呼び出し
                if self.check_convergence(curr_log_likelihood, prev_log_likelihood, tol=self.tol):
                    self.converged_ = True
                    print("Converged at iteration", iteration + 1)
                    break

                prev_log_likelihood = curr_log_likelihood

            # end of for loop ---

        # 推定パラメータの並び替え
        self._reorder_components()

        return self

    def _e_step(self, Y, X) -> None:
        """
        E ステップ：現在のパラメータを用いて、各サンプルと各クラスタの
        posterior posterior r_ij を計算する。
        戻り値: r (形状 n_samples x n_components の行列)
        各サンプル i, 各クラスタ v に対する「責任度」（posterior）を計算

        Input:
            Y
            X
            W
        Output:
        posterior : ndarray of shape (n_samples, n_components)
            各サンプル i に対する各クラスタ v の posterior 確率 r_ij
            形: r_ij = p_iv = π_v(X_i, W_i) * 𝒩_h [f_v(Y_i | X_i)] / Σ_{k} π_k * 𝒩_h [f_k(Y_i | X_i)]
            ここで、𝒩_h はカーネル密度推定の結果を表す。


        """
        # 1) log π_v
        log_pi = np.log(np.clip(self.gate.predict_proba(X), 1e-300, 1.0))  # shape (n,m)

        # 2) log f_v
        log_f_cols = []
        for v in range(self.n_components):
            log_f = self.density[v].log_conditional_density(X, Y)
            log_f_cols.append(log_f)
        log_f = np.column_stack(log_f_cols)  # shape (n,m)

        # 3) log posterior  up to additive const
        log_num = log_pi + log_f  # (n,m)
        log_den = logsumexp(log_num, axis=1, keepdims=True)  # (n,1)

        self.posterior = np.exp(log_num - log_den)  # normalised r_iv

    def _m_step(self, Y, X) -> None:
        """
        M ステップ：E ステップで得た posterior を用いて、
        混合係数と各クラスタのパラメータを更新する。
        戻り値: 更新されたパラメータ
        """
        # 1) 混合係数 π_v (X_i, W_i) の更新
        self.gate.fit(X, y_soft=self.posterior)

        # 2) 密度 f_v(Y_i | X_i) の更新
        for v in range(self.n_components):
            self.density[v].fit(X, Y, sample_weight=self.posterior[:, v])

    def compute_log_likelihood(self, Y, X) -> float:
        """
        現在のパラメータでサンプル全体の対数尤度を計算する。
        Parameters
        ----------
        Y : array-like of shape (n_samples,)
            目的変数の観測値
        X : array-like of shape (n_samples, n_features_x)
            説明変数のデータ行列
        W : array-like of shape (n_samples, n_features_w)
            追加の説明変数（必要に応じて）

        Returns
        -------
        log_likelihood : float
            対数尤度の値
        """
        # π_v(X,W)
        pi = self.gate.predict_proba(X)  # (n, m)
        log_pi = np.log(np.clip(pi, 1e-300, 1.0))  # underflow 防止

        # 2) log f_v(y|x)  各クラスタ列を構築
        log_f_cols = []
        for v in range(self.n_components):
            log_f = self.density[v].log_conditional_density(X, Y)
            log_f_cols.append(log_f)
        log_f = np.column_stack(log_f_cols)

        # 3) log-sum-exp over components, then sum over samples
        loglik = logsumexp(log_pi + log_f, axis=1).sum()
        return float(loglik)

    def check_convergence(self, curr, prev, tol):
        """
        対数尤度の変化が tol 以下であれば収束とみなす。
        prev が -inf の場合は収束しない。
        """
        if prev == -np.inf:
            return False
        return np.abs(curr - prev) / (np.abs(prev) + 1e-12) < tol

    def _component_parameter_vector(self, index: int) -> np.ndarray:
        """
        各エキスパートのパラメータベクトルを取得し、並べ替えのキーとして利用する。
        LinearModel の場合は係数ベクトル、KDE 系ではカーネルパラメータを連結する。
        """
        model = self.density[index]

        # LinearModel 等: coef_ や sigma_ が存在する場合に安全に取得して連結する
        if getattr(model, "coef_", None) is not None:
            coef = np.asarray(getattr(model, "coef_", [])).ravel()
            sigma = np.asarray(getattr(model, "sigma_", [])).ravel()

            pieces: list[np.ndarray] = []
            if coef.size:
                pieces.append(coef)
            if sigma.size:
                pieces.append(sigma)

            if pieces:
                return np.concatenate(pieces)
            # coef_ も sigma_ も空の場合のフォールバック
            raise ValueError(f"Component model at index {index} has no coef_ or sigma_ parameters.")

        # KDE 系など: kernel_params を辞書的に連結
        if getattr(model, "kernel_params", None) is not None:
            params = model.kernel_params
            pieces: list[np.ndarray] = []
            for key in sorted(params):
                value = params[key]
                if value is None:
                    pieces.append(np.array([np.nan]))
                else:
                    pieces.append(np.atleast_1d(np.asarray(value)).ravel())
            if pieces:
                return np.concatenate(pieces)
            return np.array([0.0])

        raise AttributeError(f"Component model at index {index} does not expose sortable parameters.")

    def _reorder_components(self) -> None:
        """
        エキスパートモデルを辞書順に並び替え、posterior とゲートパラメータを整合させる。
        """
        if self.n_components <= 1:
            return

        sort_keys = [tuple(self._component_parameter_vector(v)) for v in range(self.n_components)]
        permutation = np.array(
            sorted(range(self.n_components), key=lambda idx: sort_keys[idx]),
            dtype=int,
        )

        if np.array_equal(permutation, np.arange(self.n_components)):
            # 既に辞書順
            self.component_permutation_ = permutation
            return

        # posterior の列を並び替え
        if hasattr(self, "posterior"):
            self.posterior = self.posterior[:, permutation]

        # density を新しいインデックスに再割り当て
        reordered_density: dict[int, Any] = {}
        for new_idx, old_idx in enumerate(permutation):
            reordered_density[new_idx] = self.density[old_idx]
        self.density = reordered_density

        # ゲートのパラメータを再固定
        self.gate.relabel(permutation, only_order=True)

        self.component_permutation_ = permutation
