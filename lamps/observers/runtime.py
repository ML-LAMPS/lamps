import numpy as np
import gymnasium as gym

from .base import BaseObserver


class RuntimeObserver(BaseObserver):
    """
    Observation class that provides the current runtime.
    """

    NAME = "runtime"
    MAX_RUNTIME = 72 * 60 * 60.0  # 72 hours in seconds

    @property
    def epoch_counts(self) -> np.ndarray:
        return self.env.epoch_counts

    def __init__(self, env):
        super().__init__(env)

        self.runtime_prefix = [
            np.asarray(self.repository.elapsed_time_prefix[model], dtype=np.float32)
            for model in self.models
        ]

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0,
            high=self.MAX_RUNTIME,
            shape=(self.env.num_models,),
            dtype=np.float32,
        )

    def observe(self):
        return np.fromiter(
            (
                self.runtime_prefix[i][epoch] if self.runtime_prefix[i].size else 0.0
                for i, epoch in enumerate(self.epoch_counts)
            ),
            dtype=np.float32,
            count=self.num_models,
        )

    def search_time(self):
        return float(
            np.sum(
                np.fromiter(
                    (
                        (
                            self.runtime_prefix[i][epoch]
                            if self.runtime_prefix[i].size
                            else 0.0
                        )
                        for i, epoch in enumerate(self.epoch_counts)
                    ),
                    dtype=np.float32,
                    count=self.num_models,
                )
            )
        )
