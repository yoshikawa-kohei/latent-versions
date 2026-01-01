import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from experiments.pipelines.jobs.build_jobs import build_job_configs
from experiments.pipelines.simulation import MonteCarloSimulation
from experiments.pipelines.tasks.task import run_task
from experiments.utils.config_loader import load_config

DEFAULT_CONFIG = Path("experiments/configs/N_500+J_2_22+P_10+SNR_10.yaml")
DEFAULT_OUT_DIR = Path("results/example/iter_0")


def run_one_simulation(cfg: DictConfig) -> None:
    run_task(cfg)


def load_artifacts(out_dir: Path) -> dict[str, Any]:
    dataset = np.load(out_dir / "dataset.npz")
    with (out_dir / "models.pkl").open("rb") as f:
        models = pickle.load(f)
    ipw = pd.read_csv(out_dir / "ipw.csv")
    cfg = OmegaConf.load(out_dir / "config.yaml")
    return {
        "dataset": dataset,
        "models": models,
        "ipw": ipw,
        "config": cfg,
    }


def _count_missing(array: np.ndarray) -> int:
    if np.issubdtype(array.dtype, np.floating):
        return int(np.isnan(array).sum())
    return 0


def _format_ipw_stats(ipw: pd.DataFrame) -> str:
    if ipw.empty or "ipw" not in ipw.columns:
        return "ipw: n/a"
    values = ipw["ipw"].to_numpy()
    return f"ipw: min={values.min():.4f}  median={np.median(values):.4f}  max={values.max():.4f}"


def _feature_names(n_features: int, include_intercept: bool = True) -> list[str]:
    names = ["Intercept"] if include_intercept else []
    names.extend([f"X{i + 1}" for i in range(n_features)])
    return names


def _topk_rows(feature_names: Iterable[str], coef: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    coef_vec = np.asarray(coef).ravel()
    abs_coef = np.abs(coef_vec)
    order = np.argsort(abs_coef)[::-1]
    top = order[:top_k]
    return [(list(feature_names)[i], float(coef_vec[i])) for i in top]


def _format_topk_table(feature_names: list[str], coef: np.ndarray, top_k: int = 5) -> str:
    coef_vec = np.asarray(coef).ravel()
    top_rows = _topk_rows(feature_names, coef_vec, top_k=coef_vec.size)
    lines = ["feature        coef"]
    for name, value in top_rows:
        lines.append(f"{name:<12} {value:+.4f}")

    return "\n".join(lines)


def print_summary(artifacts: dict[str, Any]) -> None:
    dataset = artifacts["dataset"]
    models = artifacts["models"]
    ipw = artifacts["ipw"]
    cfg = artifacts["config"]

    X = dataset["X"]
    T = dataset["T"]
    V = dataset["V"]
    Y = dataset["Y"]

    n_samples, n_features = X.shape
    out_dir = OmegaConf.select(cfg, "job_config.out_dir")
    sim_id = OmegaConf.select(cfg, "iter")

    print("== RUN SUMMARY ==")
    print(f"dataset: n={n_samples}, p={n_features} | T levels={np.unique(T).size} | V levels={np.unique(V).size} | Y={Y.dtype}")
    print(f"missing: X={_count_missing(X)}  Y={_count_missing(Y)}  T={_count_missing(T)}  V={_count_missing(V)}")

    treatment_model = models.get("treatment_model")
    version_models = models.get("version_models", {})
    treatment_name = type(treatment_model).__name__ if treatment_model is not None else "None"
    moe_summary = "  ".join([f"t={t} (K={getattr(m, 'n_components', 'n/a')})" for t, m in sorted(version_models.items())])
    print(f"models: treatment={treatment_name} | MoE: {moe_summary if moe_summary else 'n/a'}")
    print(_format_ipw_stats(ipw))
    print(f"output: {out_dir} | sim_id={sim_id}")

    print("\n== IPW ==")
    if not ipw.empty:
        print(ipw[["treatment", "version", "ipw"]].to_string(index=False))
    else:
        print("(empty)")

    print("\n== MODEL DETAILS ==")
    if treatment_model is None:
        print("[treatment] none")
    else:
        coef = getattr(treatment_model, "coef_", None)
        print(f"[treatment] {treatment_name}")
        if coef is not None:
            coef = np.asarray(coef)
            feature_names = _feature_names(n_features, include_intercept=getattr(treatment_model, "fit_intercept", True))
            print(f"  coef shape={coef.shape}")
            for class_idx in range(coef.shape[1]):
                print(f"  class={class_idx + 1} (vs base)")
                print(_format_topk_table(feature_names, coef[:, class_idx]))
        else:
            print("  coef_: n/a")

    for treat_index, version_model in sorted(version_models.items()):
        print(f"\n[MoE] treatment={treat_index}")
        gate = getattr(version_model, "gate", None)
        if gate is not None and getattr(gate, "coef_", None) is not None:
            gate_coef = np.asarray(gate.coef_)
            print(f"  gate coef shape={gate_coef.shape}")
            gate_feature_names = _feature_names(n_features, include_intercept=getattr(gate, "fit_intercept", True))
            for class_idx in range(gate_coef.shape[1]):
                print(f"  gate class={class_idx + 1} (vs base)")
                print(_format_topk_table(gate_feature_names, gate_coef[:, class_idx]))
        else:
            print("  gate coef_: n/a")

        density = getattr(version_model, "density", {})
        expert_coefs = []
        sigma2_list = []
        for component, outcome_model in sorted(density.items()):
            coef = getattr(outcome_model, "coef_", None)
            sigma2 = getattr(outcome_model, "sigma2_", None)
            if coef is not None:
                expert_coefs.append(np.asarray(coef).ravel())
            if sigma2 is not None:
                sigma2_list.append(float(sigma2))
            print(f"  expert v={component}")
            if coef is not None:
                outcome_feature_names = _feature_names(n_features, include_intercept=getattr(outcome_model, "fit_intercept", True))
                print(_format_topk_table(outcome_feature_names, coef))
            if sigma2 is not None:
                print(f"  sigma2={sigma2}")

    print("\n== CONFIG ==")
    print(f"out_dir={OmegaConf.select(cfg, 'job_config.out_dir')}")
    print(f"number_of_simulation={OmegaConf.select(cfg, 'job_config.number_of_simulation')}")
    data_cfg = OmegaConf.select(cfg, "job_config.data")
    if data_cfg is not None:
        print("data:")
        for key in ["n_samples", "n_treatments", "n_versions", "covariate_dim", "snr"]:
            if key in data_cfg:
                print(f"  {key}={data_cfg[key]}")


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    cfg.number_of_simulation = 1
    cfg.out_dir = "results/example"

    jobs = list(build_job_configs(cfg))
    simulation = MonteCarloSimulation(run_one_simulation=run_one_simulation, n_jobs=1)
    simulation.run(jobs, parallel=False)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

    artifacts = load_artifacts(DEFAULT_OUT_DIR)
    print_summary(artifacts)
