from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """OmegaConf を用いて YAML/TOML/JSON をロードする。"""

    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config が見つかりません: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"設定ファイルは辞書型である必要があります（{cfg_path}）")

    return cfg
