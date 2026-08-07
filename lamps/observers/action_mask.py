import numpy as np
import gymnasium as gym

from .base import BaseObserver


class ActionMaskObserver(BaseObserver):
    """
    Observation class that provides the current action mask.
    """

    NAME = "action_mask"

    def __init__(self, env):
        super().__init__(env)

        self.valid_mask = self.compute_valid_mask()

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_models,),
            dtype=np.float32,
        )

    def reset(self):
        self.valid_mask = self.compute_valid_mask()

    def observe(self):
        return self.valid_mask.astype(np.float32)

    def step(self, action):
        self.valid_mask = self.compute_valid_mask()

    def compute_valid_mask(self):
        return self.env.epoch_counts < self.env.available_epochs
