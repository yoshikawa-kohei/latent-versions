from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array, check_consistent_length, check_X_y
from tqdm.auto import tqdm


# PyTorchモデルの定義
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super(LogisticRegressionModel, self).__init__()

        if n_features <= 0:
            raise ValueError("特徴量の次元数は1以上である必要があります")

        self.n_features = n_features
        self.n_classes = n_classes

        # クラス数が1の場合は学習パラメータを持たない
        if n_classes <= 1:
            self.has_params = False
        else:
            self.has_params = True
            # 最初のクラスを除いた n_classes - 1 個のクラスに対してパラメータを学習
            self.linear = nn.Linear(n_features, n_classes - 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多項ロジットモデルの順伝播

        Args:
            x: 入力特徴量 [batch_size, n_features]

        Returns:
            logits: 各クラスの対数確率 [batch_size, n_classes]
        """
        batch_size = x.size(0)
        # 入力の特徴量次元を確認
        if x.size(1) != self.n_features:
            raise ValueError(f"入力の特徴量次元 {x.size(1)} が初期化時の次元 {self.n_features} と一致しません")

        # クラス数が1の場合は単にゼロベクトルを返す
        if not self.has_params:
            return torch.zeros(batch_size, 1, device=x.device)

        # n_classes - 1 個のクラスに対する線形変換を計算
        reduced_logits = self.linear(x)

        # 最初のクラスに対応する0ベクトルを追加
        zero_logits = torch.zeros(batch_size, 1, device=x.device)

        # 全クラスの出力を結合
        return torch.cat([zero_logits, reduced_logits], dim=1)


class MultinomialLogisticRegression:
    """
    ハードラベル・ソフトラベル対応の多項ロジスティック回帰分類器

    Parameters
    ----------
    optimizer : str, default='lbfgs'
        最適化アルゴリズム ('lbfgs', 'adam', 'sgd', 'rmsprop', 'adagrad')
    lr : float, default=0.01
        学習率
    max_iter : int, default=1000
        最大イテレーション数
    batch_size : int, default=32
        バッチサイズ
    fit_intercept : bool, default=True
        切片を含めるかどうか
    tol : float, default=1e-4
        収束判定の閾値
    weight_decay : float, default=0.0
        L2正則化の強さ
    random_state : int, default=None
        乱数シード
    device : str, default=None
        計算に使用するデバイス ('cpu' または 'cuda')
        None の場合は自動的に利用可能なデバイスを選択
    early_stopping : bool, default=False
        早期終了を使用するかどうか
    patience : int, default=10
        早期終了の判定に使用する我慢回数
    """

    def __init__(
        self,
        optimizer: str = "adam",
        lr: float = 0.01,
        max_iter: int = 1000,
        batch_size: int = 1024,
        fit_intercept: bool = True,
        tol: float = 1e-4,
        weight_decay: float = 0.0,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
        early_stopping: bool = False,
        patience: int = 10,
        num_workers: int = 2,
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.num_workers = num_workers

        # デバイスの設定
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

        self.model = None
        self.classes_ = None
        self.history_: Dict[str, List[float]] = {"loss": [], "val_loss": []}

        self.is_fitted = False

    def _set_random_state(self) -> None:
        """乱数シードを設定"""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)
            # 再現性のために決定論的アルゴリズムを使用
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """特徴量行列に切片用の列を追加"""
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate([intercept, X], axis=1)
        return X

    def _get_optimizer(self, parameters: Any) -> torch.optim.Optimizer:
        """オプティマイザを取得する"""
        # 最適化アルゴリズムの選択
        if self.optimizer.lower() == "lbfgs":
            return optim.LBFGS(
                parameters,
                max_iter=self.max_iter,
                line_search_fn="strong_wolfe",
                tolerance_grad=self.tol,
                tolerance_change=self.tol,
            )
        elif self.optimizer.lower() == "sgd":
            return optim.SGD(parameters, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "rmsprop":
            return optim.RMSprop(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "adagrad":
            return optim.Adagrad(parameters, lr=self.lr, weight_decay=self.weight_decay)
        else:  # デフォルトはadam
            return optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)

    def _create_data_loader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, sample_weight: Optional[np.ndarray] = None, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """データローダーを作成する"""
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        if sample_weight is not None:
            w_tensor = torch.from_numpy(sample_weight).float()
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, w_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        y_soft: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        y_val_soft: Optional[np.ndarray] = None,
        sample_waight: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "MultinomialLogisticRegression":
        """
        分類器の学習を行う

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            学習データの特徴量
        y : array-like, shape (n_samples,), optional
            学習データのラベル（ハードラベル）
        y_soft : array-like, shape (n_samples, n_classes), optional
            学習データのソフトラベル
        X_val : array-like, shape (n_val_samples, n_features), optional
            検証データの特徴量
        y_val : array-like, shape (n_val_samples,), optional
            検証データのラベル（ハードラベル）
        y_val_soft : array-like, shape (n_val_samples, n_classes), optional
            検証データのソフトラベル
        verbose : bool, default=True
            学習の進捗を表示するかどうか

        Returns
        -------
        self : MultinomialLogisticRegression
            学習済みの分類器
        """
        # 入力チェック
        if y is None and y_soft is None:
            raise ValueError("Either y or y_soft must be provided")

        # 乱数シードの設定
        self._set_random_state()

        # 特徴量の処理
        X = check_array(X, ensure_2d=True, dtype=np.float32)
        X_with_intercept = self._add_intercept(X)
        n_samples, n_features = X_with_intercept.shape

        # ラベルの処理
        if y_soft is not None:
            # ソフトラベルの場合
            y_labels = check_array(y_soft, ensure_2d=True, dtype=np.float32)
            check_consistent_length(X, y_labels)
            n_classes = y_labels.shape[1]
            self.classes_ = np.arange(n_classes)
            self.use_soft_labels = True
        else:
            # ハードラベルの場合
            X, y = check_X_y(X, y, ensure_2d=True, dtype=np.float32)
            # OneHotEncoding
            self.encoder_ = OneHotEncoder(sparse_output=False)
            y_labels = self.encoder_.fit_transform(y.reshape(-1, 1))
            self.classes_ = self.encoder_.categories_[0]
            n_classes = len(self.classes_)
            self.use_soft_labels = False

        # 検証データの処理
        has_validation = X_val is not None and (y_val is not None or y_val_soft is not None)
        if has_validation:
            X_val = check_array(X_val, ensure_2d=True, dtype=np.float32)
            X_val_with_intercept = self._add_intercept(X_val)

            if y_val_soft is not None:
                y_val_labels = check_array(y_val_soft, ensure_2d=True, dtype=np.float32)
                check_consistent_length(X_val, y_val_labels)
            else:
                X_val, y_val = check_X_y(X_val, y_val, ensure_2d=True, dtype=np.float32)
                y_val_labels = self.encoder_.transform(y_val.reshape(-1, 1))

        # モデルの初期化
        self.model = LogisticRegressionModel(n_features, n_classes)

        if n_classes <= 1:
            return self

        optimizer = self._get_optimizer(self.model.parameters())

        # 損失関数
        criterion = nn.CrossEntropyLoss(reduction="none") if not self.use_soft_labels else None

        # データローダーを作成
        train_loader = self._create_data_loader(X_with_intercept, y_labels, self.batch_size, shuffle=True)

        if has_validation:
            val_loader = self._create_data_loader(X_val_with_intercept, y_val_labels, self.batch_size, shuffle=False)

        # 早期終了の設定
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        # 学習ループ
        self.model.train()
        pbar = tqdm(range(self.max_iter), disable=not verbose)
        for epoch in pbar:
            # トレーニング
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            self.history_["loss"].append(train_loss)

            # 検証
            if has_validation:
                val_loss = self._validate(val_loader, criterion)
                self.history_["val_loss"].append(val_loss)

                # 進捗バーの更新
                if verbose:
                    pbar.set_description(f"Epoch {epoch + 1}/{self.max_iter} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

                # 早期終了の判定
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            if verbose:
                                print(f"Early stopping triggered at epoch {epoch + 1}")
                            # 最良のモデルを復元
                            self.model.load_state_dict(best_model_state)
                            break
            else:
                # 進捗バーの更新（検証データなし）
                if verbose:
                    pbar.set_description(f"Epoch {epoch + 1}/{self.max_iter} - loss: {train_loss:.4f}")

            # 収束判定
            if len(self.history_["loss"]) > 1:
                if abs(self.history_["loss"][-1] - self.history_["loss"][-2]) < self.tol:
                    if verbose:
                        print(f"Converged at epoch {epoch + 1}")
                    break

        # 学習済みのパラメータを取得（sklearn互換の属性として保存）
        self.coef_ = self.model.linear.weight.data.cpu().numpy().T
        # if self.fit_intercept:
        #     self.intercept_ = self.model.linear.bias.data.cpu().numpy()

        self.is_fitted = True

        return self

    def _train_epoch(self, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: Optional[nn.Module]) -> float:
        """1エポックの学習を行う"""
        self.model.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            if len(batch) == 3:
                batch_X, batch_y, batch_w = batch
            else:
                batch_X, batch_y = batch
                batch_w = None

            batch_size = batch_X.size(0)
            total_samples += batch_size

            # 勾配を初期化
            optimizer.zero_grad()

            # 順伝播
            outputs = self.model(batch_X)

            # ロス計算
            if self.use_soft_labels:
                # ソフトラベルの場合はクロスエントロピーを手動で計算
                # log_softmax = nn.LogSoftmax(dim=1)(outputs)
                # log_softmax = F.log_softmax(outputs, dim=1)  # shape = [B, K]

                # # loss = -torch.sum(batch_y * log_softmax) / batch_size
                # per_sample = -torch.sum(batch_y * log_softmax, dim=1)
                # if batch_w is not None:
                #     loss = (per_sample * batch_w).sum() / batch_size
                # else:
                #     loss = per_sample.mean()

                # 1) インスタンス生成を廃止し F.log_softmax を利用
                logp = F.log_softmax(outputs, dim=1)  # shape = [B, K]

                # 2) 行ごとの内積を einsum / bmm で一括計算（メモリ削減）
                loss_vec = torch.einsum("bk,bk->b", batch_y, -logp)  # shape = [B]

                # 3) サンプル重みが無ければ単純平均
                loss = loss_vec.mean() if batch_w is None else (loss_vec * batch_w).mean()

            else:
                # ハードラベルの場合は通常のクロスエントロピー
                # loss = criterion(outputs, torch.argmax(batch_y, dim=1))
                per_sample = criterion(outputs, torch.argmax(batch_y, dim=1))  # shape=(batch,)
                if batch_w is not None:
                    loss = (per_sample * batch_w).sum() / batch_size
                else:
                    loss = per_sample.mean()

            # 逆伝播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size

        return epoch_loss / total_samples

    def _validate(self, dataloader: torch.utils.data.DataLoader, criterion: Optional[nn.Module]) -> float:
        """検証データで評価を行う"""
        self.model.eval()
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    batch_X, batch_y, batch_w = batch
                else:
                    batch_X, batch_y = batch
                    batch_w = None

                batch_size = batch_X.size(0)
                total_samples += batch_size

                # 順伝播
                outputs = self.model(batch_X)

                # ロス計算
                if self.use_soft_labels:
                    # ソフトラベルの場合はクロスエントロピーを手動で計算
                    log_softmax = nn.LogSoftmax(dim=1)(outputs)
                    # loss = -torch.sum(batch_y * log_softmax) / batch_size
                    per_sample = -torch.sum(batch_y * log_softmax, dim=1)
                    if batch_w is not None:
                        loss = (per_sample * batch_w).sum() / batch_size
                    else:
                        loss = per_sample.mean()
                else:
                    # ハードラベルの場合は通常のクロスエントロピー
                    # loss = criterion(outputs, torch.argmax(batch_y, dim=1))
                    per_sample = criterion(outputs, torch.argmax(batch_y, dim=1))  # shape=(batch,)
                    if batch_w is not None:
                        loss = (per_sample * batch_w).sum() / batch_size
                    else:
                        loss = per_sample.mean()

                    val_loss += loss.item() * batch_size

        return val_loss / total_samples

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            予測したいデータの特徴量

        Returns
        -------
        array-like, shape (n_samples, n_classes)
            各クラスの予測確率
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() before predict_proba().")

        X = check_array(X, ensure_2d=True, dtype=np.float32)
        X_with_intercept = self._add_intercept(X)

        # 予測
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_with_intercept).float()
            logits = self.model(X_tensor)
            probas = nn.Softmax(dim=1)(logits)

        return probas.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        クラス予測を行う

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            予測したいデータの特徴量

        Returns
        -------
        array-like, shape (n_samples,)
            予測クラス
        """
        proba = self.predict_proba(X)
        y_pred_index = np.argmax(proba, axis=1)
        return self.classes_[y_pred_index]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        モデルの精度を計算する

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            テストデータの特徴量
        y : array-like, shape (n_samples,)
            テストデータのラベル

        Returns
        -------
        float
            精度(accuracy)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def relabel(self, permutation: Sequence[int], only_order=False) -> None:
        """
        指定した順序にクラスを並べ替え、基準クラスの係数を0に再固定する。
        """
        if permutation is None:
            return

        perm = np.asarray(permutation, dtype=int)

        if self.classes_ is None:
            raise RuntimeError("classes_ is not set")

        n_classes = len(self.classes_)
        if perm.shape[0] != n_classes:
            raise ValueError("Permutation length must match number of classes.")
        if set(perm.tolist()) != set(range(n_classes)):
            raise ValueError("Permutation must be a rearrangement of class indices.")

        if n_classes <= 1:
            # 1クラス以下なら並べ替えだけ
            self.classes_ = np.array(self.classes_)[perm]
            return

        if not hasattr(self, "coef_"):
            raise RuntimeError("coef_ is required for relabel")

        # 現在の coef_ は形状 (n_features, n_classes-1)
        # まず全クラス分の係数行列を構築（基準クラス = 最初の列がゼロ）
        full_coef = np.zeros((self.coef_.shape[0], n_classes), dtype=self.coef_.dtype)
        full_coef[:, 1:] = self.coef_

        # 新しい基準クラスを perm[0] にし、そのクラスをゼロに再固定するために差を引く
        base_idx = perm[0]
        base_vec = full_coef[:, base_idx].copy()
        rebased = full_coef - base_vec[:, None]
        rebased[:, base_idx] = 0.0

        # 指定順に並べ替え、再び基準クラス列を除いて coef_ を更新
        ordered = rebased[:, perm]
        new_coef = ordered[:, 1:].astype(self.coef_.dtype, copy=False)

        self.coef_ = new_coef

        # PyTorch モデルの重みも更新
        with torch.no_grad():
            device = self.model.linear.weight.device
            weight_tensor = torch.from_numpy(new_coef.T).to(device=device, dtype=self.model.linear.weight.dtype)
            self.model.linear.weight.copy_(weight_tensor)

        # classes_ を指定順に並べ替えて保存

        if only_order:
            # 潜在変数を使う場合はラベルに意味はないため，こちらを通る
            return
        else:
            self.classes_ = np.array(self.classes_)[perm]

    def log_likelihood(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> float:
        """
        対数尤度 (log-likelihood) を計算する

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            データの特徴量
        y : array-like, shape (n_samples, n_classes)
            ラベル

        Returns
        -------
        float
            対数尤度
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() before computing likelihood.")

        # 確率の取得
        probas = self.predict_proba(X)  # shape (n_samples, n_classes)

        # 対数確率
        log_probas = np.log(probas + 1e-10)  # 数値安定化のため微小定数を足す

        if self.use_soft_labels:
            # ソフトラベルの場合
            return np.sum(y * log_probas)
        else:
            # ハードラベルの場合
            y_onehot = self.encoder_.transform(y.reshape(-1, 1))

            return np.sum(y_onehot * log_probas)
