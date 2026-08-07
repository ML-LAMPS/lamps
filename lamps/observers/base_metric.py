import numpy as np
import gymnasium as gym

from .base import BaseObserver


class BaseMetricObserver(BaseObserver):
    """
    Observation class that provides the current eval/log_loss for each model.
    """

    NAME = "metric_name"

    def __init__(self, env):
        super().__init__(env)

        self.log_loss_arrays = [
            np.asarray(self.repository.data[model][self.NAME], dtype=np.float32)
            for model in self.models
        ]

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-15.0,
            high=15.0,
            shape=(self.num_models,),
            dtype=np.float32,
        )

    def observe(self):
        return np.fromiter(
            (
                self.log_loss_arrays[i][epoch]
                for i, epoch in enumerate(self.env.epoch_counts)
            ),
            dtype=np.float32,
            count=self.num_models,
        )
