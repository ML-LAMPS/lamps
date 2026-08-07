import numpy as np
import gymnasium as gym

from .base_metric import BaseMetricObserver


class EvalLogLossObserver(BaseMetricObserver):
    """
    Observation class that provides the current eval/log_loss for each model.
    """

    NAME = "eval/log_loss"

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-15.0,
            high=15.0,
            shape=(self.num_models,),
            dtype=np.float32,
        )
