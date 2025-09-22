from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import yaml


def configure_logging(level: str = "INFO", *, fmt: str | None = None) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt or "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("tft")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"tft.{name}")


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_json(obj: Any, path: Path | str, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=indent)


def ensure_keys(d: dict[str, Any], chain: Iterable[str]) -> None:
    cur: Any = d
    seen: list[str] = []
    for k in chain:
        seen.append(k)
        if k not in cur:
            raise KeyError(f"Missing config key: {'.'.join(seen)}")
        cur = cur[k]


def coerce_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise ValueError(f"Cannot convert {type(obj)} to dict")


def load_best_hyperparameters(results_path: str | Path) -> dict[str, Any]:
    logger = get_logger("utils")
    with Path(results_path).open("r") as f:
        ray_results = json.load(f)
    optimal_config: dict[str, Any] = ray_results["best_params"]

    logger.info("Best parameters from Ray Tune:")
    for key, value in optimal_config.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")

    return optimal_config
