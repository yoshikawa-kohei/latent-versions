import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from causal_versions.estimator.mnlogit import MultinomialLogisticRegression
from causal_versions.estimator.MoE import MoE
from experiments.utils import GeneratorConfig, SyntheticDataGenerator

DATASET_FILENAME = "dataset.npz"
MODELS_FILENAME = "models.pkl"
IPW_FILENAME = "ipw.csv"
CONFIG_FILENAME = "config.yaml"


def generate_dataset(cnf: DictConfig, seed):
    generator_cfg = GeneratorConfig(
        n_samples=cnf.n_samples,
        n_treatments=cnf.n_treatments,
        n_versions=cnf.n_versions,
        covariate_dim=cnf.covariate_dim,
        treatment_strength=cnf.treatment_strength,
        version_strength=cnf.version_strength,
        outcome_snr=cnf.snr,
        seed=int(seed),
    )
    generator = SyntheticDataGenerator(generator_cfg)
    X, T, V, Y = generator.generate()
    # XW = np.concatenate([X, W], axis=1)

    return X, T, V, Y


def fit_model(cnf: DictConfig, dataset):
    X, T, V, Y = dataset

    n_treatments = len(np.unique(T))

    for t in range(n_treatments):
        print(f"   t={t}  V counts:", np.bincount(V[T == t]))

    # training treatment model
    treatment_model = MultinomialLogisticRegression()
    treatment_model.fit(X=X, y=T)

    # fit version models
    version_models = {}
    for t in range(n_treatments):
        mask = T == t
        X_t, Y_t, _ = X[mask], Y[mask], V[mask]

        version_cnf = cnf.version
        version_model = MoE(**version_cnf.kwargs)
        version_model.fit(Y_t, X_t)

        version_models[t] = version_model

    return treatment_model, version_models


def calculate_ipw(dataset, models):
    X, T, V, Y = dataset
    treatment_model, version_models = models
    n_treatments = len(np.unique(T))
    n_samples = X.shape[0]

    results = []
    for t in range(n_treatments):
        mask = T == t
        X_t, Y_t, V_t = X[mask], Y[mask], V[mask]
        ipw = calculate_causal_effect(t, treatment_model, version_models[t], Y_t, X_t, n_samples)
        results.append(ipw)

    return results


def calculate_causal_effect(treat_index, treatment_model: MultinomialLogisticRegression, version_model: MoE, Y, X: np.ndarray, n_samples: int):
    # 因果効果
    # ----------------------------
    # ① 処置 t サンプルに対する処置割当確率 e_t(X)
    e_t = treatment_model.predict_proba(X)  # (n_samples, n_treatments)

    # ② 処置 t サンプルに対するバージョン割当確率 π_{t,v}(X)
    pi_tv = version_model.gate.predict_proba(X)  # gateの予測確率を計算  (n_samples, n_components)

    # ③ 処置 t に属する行番号・サブセット
    posterior_t = version_model.posterior  # (n_t, K_t)

    # IPW の計算
    # ④ ψ̂_{t,v} を計算
    # treatmentの処置確率 x バージョンの確率 でアウトカム Y を逆確率重みづけする
    # IPW = \frac{1}{n} \sum_i=1^n Y_i * \frac{I(T_i = t) * posterior_i}{treatment_probs_i[t] * version_probs_i[t,v]}
    ipw_hat = {}

    # IPW
    for v in range(posterior_t.shape[1]):
        weights = posterior_t[:, v] / (e_t[:, treat_index] * pi_tv[:, v])  # (n_t,)
        # ipw_hat[(treat_index, v)] = np.sum(Y * weights) / np.sum(weights)  # Hajek
        ipw_hat[(treat_index, v)] = np.sum(Y * weights) / n_samples  # HT

    return ipw_hat


def run_task(cnf: DictConfig):
    iteration: int = cnf.iter
    job_config: DictConfig = cnf.job_config

    job_label = f"iter={iteration}"

    # ------- run simulation -------
    print(f"[run] {job_label}")
    dataset = generate_dataset(job_config.data, seed=iteration)
    models = fit_model(job_config.model, dataset)
    ipw = calculate_ipw(dataset, models)
    print(f"[done] {job_label}")

    # ------- save simulation -------
    out_dir = Path(job_config.out_dir) / f"iter_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_dataset(dataset, out_dir / DATASET_FILENAME)
    _save_models(models, out_dir / MODELS_FILENAME)
    _save_ipw(ipw, out_dir / IPW_FILENAME)

    OmegaConf.save(cnf, out_dir / CONFIG_FILENAME)
    print(f"[save] artifacts -> {out_dir}")

    return cnf, dataset, ipw


def _save_dataset(dataset: tuple[np.ndarray, ...], path: Path) -> None:
    arrays = {
        "X": dataset[0],
        "T": dataset[1],
        "V": dataset[2],
        "Y": dataset[3],
    }
    np.savez_compressed(path, **arrays)


def _save_models(models: tuple[MultinomialLogisticRegression, dict[int, MoE]], path: Path) -> None:
    treatment_model, version_models = models
    _move_model_to_cpu(getattr(treatment_model, "model", None))
    for version_model in version_models.values():
        gate = getattr(version_model, "gate", None)
        _move_model_to_cpu(getattr(gate, "model", None))
    payload = {
        "treatment_model": treatment_model,
        "version_models": version_models,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _move_model_to_cpu(torch_model: Any) -> None:
    if torch_model is not None and hasattr(torch_model, "to"):
        torch_model.to("cpu")


def _save_ipw(ipw_results: list[dict[tuple[int, int], float]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for entry in ipw_results:
        for (treat, version), value in entry.items():
            rows.append({"treatment": int(treat), "version": int(version), "ipw": float(value)})

    df = pd.DataFrame(rows, columns=["treatment", "version", "ipw"])
    if not df.empty:
        df = df.sort_values(["treatment", "version"]).reset_index(drop=True)
    df.to_csv(path, index=False)
