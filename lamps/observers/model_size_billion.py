import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import gymnasium as gym

from .base import BaseObserver


@lru_cache(maxsize=1)
def _load_model_sizes() -> dict[str, float]:
    root = Path(__file__).resolve().parents[2]
    with open(root / "model_sizes.json", "r", encoding="utf-8") as f:
        return json.load(f)


class ModelSizeBillionObserver(BaseObserver):
    """
    Observation class that provides each model size in billions of parameters.
    """

    NAME = "model/size_billion"

    def __init__(self, env):
        super().__init__(env)

        model_sizes = _load_model_sizes()
        self.model_sizes_billion = np.asarray(
            [model_sizes[model] / 1e9 for model in self.models],
            dtype=np.float32,
        )

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0.0,
            high=10.0,
            shape=(self.num_models,),
            dtype=np.float32,
        )

    def observe(self):
        return self.model_sizes_billion
