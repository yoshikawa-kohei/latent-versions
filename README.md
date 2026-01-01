# latent-versions

Statistical methods for causal inference under multiple versions of treatment, with a lightweight Monte Carlo simulation pipeline.

## Quickstart

Prerequisites: Python `>=3.10` and `uv` (recommended).

```bash
uv sync
uv run main.py
```


## Monte Carlo Simulations

Run with multiprocessing:

```bash
uv run -m experiments.pipelines.runner --config experiments/configs/N_500+J_2_22+P_10+SNR_10.yaml --jobs 4
```

Sequential run (no multiprocessing):

```bash
uv run -m experiments.pipelines.runner --config experiments/configs/N_500+J_2_22+P_10+SNR_10.yaml --sequential
```

## What’s inside

This repository currently focuses on a simulation workflow:

- Generate synthetic data `(X, T, V, Y)` where `T` is treatment and `V` is the version within treatment.
- Fit a treatment assignment model (`MultinomialLogisticRegression`) and per-treatment version models (`MoE`).
- Compute inverse probability weighted (IPW) estimates by treatment-version pairs.

Main entrypoint: `experiments/pipelines/runner.py`.

## Configuration

Simulation runs are configured via YAML files under `experiments/configs/`.

Example (`experiments/configs/N_500+J_2_22+P_10+SNR_10.yaml`):

```yaml
out_dir: results/N_500+J_2_22+P_10+SNR_10
number_of_simulation: 100
data:
  n_samples: 500
  n_treatments: 2
  n_versions: [2, 2]
  covariate_dim: 10
  treatment_strength: 2
  version_strength: 2
  snr: 10
  binomial: false
model:
  treatment:
    cls: MultinomialLogisticRegression
    kwargs: {}
  version:
    cls: MoE
    kwargs:
      n_components: 2
      max_iter: 500
      tol: 1.0e-05
```

Config generation helper:

```bash
uv run python experiments/configs/generate_configs.py
```

## Outputs

Each Monte Carlo iteration writes artifacts to a subdirectory under `out_dir`:

- `dataset.npz`: arrays `X`, `T`, `V`, `Y`
- `models.pkl`: fitted treatment model + version models
- `ipw.csv`: IPW estimates with columns `treatment`, `version`, `ipw`
- `config.yaml`: the resolved job config used for that iteration

## Repository layout

- `causal_versions/`: estimators and modeling utilities
  - `causal_versions/estimator/mnlogit/`: multinomial logistic regression (PyTorch-backed)
  - `causal_versions/estimator/MoE/`: mixture-of-experts for version modeling
  - `causal_versions/estimator/linear/`: weighted linear regression used in MoE experts
- `experiments/`: simulation & synthetic data generator
  - `experiments/utils/generator.py`: synthetic data generation
  - `experiments/pipelines/`: runner + job definitions
  - `experiments/configs/`: YAML configs


## License

MIT (see `LICENSE`).
