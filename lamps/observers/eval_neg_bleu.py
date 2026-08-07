import numpy as np
import gymnasium as gym

from .base_metric import BaseMetricObserver


class EvalNegBleuObserver(BaseMetricObserver):
    """
    Observation class that provides the current eval/neg_bleu for each model.
    """

    NAME = "eval/neg_bleu"

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-100.0,
            high=0.0,
            shape=(self.num_models,),
            dtype=np.float32,
        )
