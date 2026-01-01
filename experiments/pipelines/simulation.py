from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Sequence

from omegaconf import DictConfig, OmegaConf

JobIterable = Iterable[DictConfig | dict[str, Any]]
SimulationFunc = Callable[[DictConfig], Any]


class MonteCarloSimulation:
    """ジョブ集合（DictConfig の列）を実行するだけの軽量ランナー。"""

    def __init__(self, run_one_simulation: SimulationFunc, n_jobs: int = 1) -> None:
        self.run_one_simulation = run_one_simulation
        self.n_jobs = max(1, n_jobs)

    # ------------------------------------------------------------------ #
    def run(self, jobs: JobIterable, parallel: bool = True) -> list[Any]:
        configs = [self._ensure_config(job) for job in jobs]
        if not configs:
            return []

        if parallel and self.n_jobs > 1:
            return self._run_parallel(configs)

        return [self.run_one_simulation(cfg) for cfg in configs]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_config(job: DictConfig | dict[str, Any]) -> DictConfig:
        if isinstance(job, DictConfig):
            return job
        return OmegaConf.create(job)

    # ------------------------------------------------------------------ #
    def _run_parallel(self, configs: Sequence[DictConfig]) -> list[Any]:
        ctx = mp.get_context("spawn")
        results: list[Any] = []

        with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=ctx) as executor:
            futures = [
                executor.submit(_job_worker, self.run_one_simulation, OmegaConf.to_container(cfg, resolve=True) or {})
                for cfg in configs
            ]
            for future in as_completed(futures):
                results.append(future.result())
        return results


def _job_worker(func: SimulationFunc, payload: dict[str, Any]) -> Any:
    cfg = OmegaConf.create(payload)
    return func(cfg)
