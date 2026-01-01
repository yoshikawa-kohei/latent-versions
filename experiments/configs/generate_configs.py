#!/usr/bin/env python3
"""Generate simulation YAML configs for all parameter combinations."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parent

N_SAMPLES = (500, 1000, 2000)
COVARIATE_DIMS = (10, 20)
SNR_VALUES = (5, 10)
VERSION_STRUCTURES: dict[int, tuple[tuple[int, ...], ...]] = {
    2: (
        (2, 2),
        (3, 3),
    ),
    3: (
        (2, 2, 2),
        (3, 3, 3),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YAML configs for simulations.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files if they exist.")
    parser.add_argument("--dry-run", action="store_true", help="Only print which files would be written.")
    return parser.parse_args()


def iter_specifications() -> Iterable[tuple[int, Sequence[int], int, int]]:
    for n_samples, covariate_dim, snr in itertools.product(N_SAMPLES, COVARIATE_DIMS, SNR_VALUES):
        for structures in VERSION_STRUCTURES.values():
            for n_versions in structures:
                yield n_samples, n_versions, covariate_dim, snr


def build_file_stem(n_samples: int, n_versions: Sequence[int], covariate_dim: int, snr: int) -> str:
    n_treatments = len(n_versions)
    version_suffix = "".join(str(v) for v in n_versions)
    return f"N_{n_samples}+J_{n_treatments}_{version_suffix}+P_{covariate_dim}+SNR_{snr}"


def build_config_dict(
    n_samples: int,
    n_versions: Sequence[int],
    covariate_dim: int,
    snr: int,
    file_stem: str,
) -> dict[str, object]:
    n_treatments = len(n_versions)
    return {
        "name": "correct_spec",
        "out_dir": f"results/{file_stem}",
        "number_of_simulation": 100,
        "data": {
            "n_samples": n_samples,
            "n_treatments": n_treatments,
            "n_versions": list(n_versions),
            "covariate_dim": covariate_dim,
            "treatment_strength": 2,
            "version_strength": 2,
            "snr": snr,
            "binomial": False,
        },
        "model": {
            "treatment": {
                "cls": "MultinomialLogisticRegression",
                "kwargs": {},
            },
            "version": {
                "cls": "MoE",
                "kwargs": {
                    "n_components": n_versions[0],
                    "max_iter": 500,
                    "tol": 1e-5,
                },
            },
        },
    }


def write_config(path: Path, config_dict: dict[str, object]) -> None:
    conf = OmegaConf.create(config_dict)
    OmegaConf.save(config=conf, f=str(path))


def main() -> None:
    args = parse_args()
    for n_samples, n_versions, covariate_dim, snr in iter_specifications():
        file_stem = build_file_stem(n_samples, n_versions, covariate_dim, snr)
        output_path = CONFIG_DIR / f"{file_stem}.yaml"
        if output_path.exists() and not args.force:
            print(f"Skip {output_path.name}: file already exists.")
            continue
        if args.dry_run:
            print(f"Would write {output_path.name}")
            continue
        config_dict = build_config_dict(n_samples, n_versions, covariate_dim, snr, file_stem)
        write_config(output_path, config_dict)
        print(f"Wrote {output_path.name}")


if __name__ == "__main__":
    main()
