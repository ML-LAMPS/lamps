import numpy as np
import gymnasium as gym

from .base import BaseObserver


class EpochObserver(BaseObserver):
    """
    Observation class that provides the current epoch count.
    """

    NAME = "epochs"
    NUM_MAX_EPOCHS = 1000

    def __init__(self, env):
        super().__init__(env)

        self.initialize()

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0,
            high=self.NUM_MAX_EPOCHS,
            shape=(self.env.num_models,),
            dtype=np.float32,
        )

    def reset(self):
        self.initialize()

    def observe(self):
        return self.epoch_counts

    def info(self):
        return {"epoch_counts": self.epoch_counts}

    def step(self, action):
        model_idx = int(action)
        self.epoch_counts[model_idx] += 1

    def initialize(self):
        self.epoch_counts = np.zeros(self.env.num_models, dtype=np.int32)
