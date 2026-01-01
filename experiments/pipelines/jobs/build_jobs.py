from __future__ import annotations

from typing import Iterable

from omegaconf import DictConfig, OmegaConf


def build_job_configs(cnf: DictConfig) -> Iterable[DictConfig]:
    """設定ファイルから Monte Carlo 用のジョブ設定列を生成する。"""

    job_configs = []
    for iter in range(cnf.number_of_simulation):
        job_config = {"iter": iter, "job_config": cnf}
        job_configs.append(OmegaConf.create(job_config))

    return job_configs
