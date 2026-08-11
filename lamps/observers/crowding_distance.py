import numpy as np
import gymnasium as gym

from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function

from .base import BaseObserver

CROWDING_DISTANCE_CAP = 100.0

crowding_function = get_crowding_function("cd")


class CrowdingDistanceObserver(BaseObserver):
    """
    Per-model NSGA-II crowding distance (Deb et al., 2002) over each model's
    current objective-space state, measuring how isolated a model is relative
    to its neighbors. Boundary/extreme models get an (clipped) infinite score.
    """

    NAME = "multi_objective/crowding_distance"

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0.0,
            high=CROWDING_DISTANCE_CAP,
            shape=(self.num_models,),
            dtype=np.float32,
        )

    def observe(self):
        datapoints = self.repository.datapoints(self.env.epoch_counts)
        distances = crowding_function.do(datapoints)
        return np.clip(distances, 0.0, CROWDING_DISTANCE_CAP).astype(np.float32)
