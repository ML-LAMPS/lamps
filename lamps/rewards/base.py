from abc import ABC, abstractmethod

import numpy as np


class BaseReward(ABC):
    """
    Base class for reward functions in the LAMPS environment.
    """

    NAME: str = None

    def __init__(self, env):
        self.env = env

    def reset(self):
        """
        Reset any per-episode state. Can be overridden by subclasses.
        """

    def before_step(self, action):
        """
        Hook called before observers mutate state for this action.
        """

    @abstractmethod
    def compute(self, terminated: bool, valid_mask: np.ndarray | None = None) -> float:
        """
        Compute the reward for the transition that just occurred.
        """

    def info(self) -> dict | None:
        """
        Return additional diagnostic info about the reward. Can be overridden.
        """
        return None
