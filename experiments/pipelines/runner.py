from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from experiments.pipelines.jobs.build_jobs import build_job_configs
from experiments.pipelines.simulation import MonteCarloSimulation
from experiments.pipelines.tasks.task import run_task
from experiments.utils.config_loader import load_config


def run_one_simulation(cfg: DictConfig) -> dict[str, Any]:
    """実際のシミュレーション処理を後で差し替えやすくするためのプレースホルダ。"""

    run_task(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo simulation runner")
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/N_500+J_2_22+P_10+SNR_10.yaml"), help="設定ファイル")
    parser.add_argument("--jobs", type=int, default=4, help="並列プロセス数")
    parser.add_argument("--sequential", action="store_true", help="逐次モードを強制")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_config(args.config)

    jobs = list(build_job_configs(experiment))
    simulation = MonteCarloSimulation(run_one_simulation=run_one_simulation, n_jobs=args.jobs)
    results = simulation.run(jobs, parallel=not args.sequential)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
