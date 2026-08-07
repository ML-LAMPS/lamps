import numpy as np
import gymnasium as gym

from .base_metric import BaseMetricObserver


class EvalNegF1MacroObserver(BaseMetricObserver):
    """
    Observation class that provides the current eval/neg_f1_macro for each model.
    """

    NAME = "eval/neg_f1_macro"

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-1.0,
            high=0.0,
            shape=(self.num_models,),
            dtype=np.float32,
        )
