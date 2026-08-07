from abc import ABC, abstractmethod

from gymnasium.spaces import Space


class BaseObserver(ABC):
    """
    Base class for observations in the LAMPS environment.
    """

    NAME: str = None

    @property
    def repository(self):
        return self.env.repository

    @property
    def models(self):
        return self.env.models

    @property
    def num_models(self):
        return self.env.num_models

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def observe(self):
        """
        Method to perform the observation. Should be implemented by subclasses.
        """

    @abstractmethod
    def to_space(self) -> Space:
        """
        Method to convert the observation to a Gymnasium space. Should be implemented by subclasses.
        """

    def reset(self):
        """
        Method to reset the observation. Should be implemented by subclasses.
        """

    def step(self, action):
        """
        Method to update the observation based on the action taken. Should be implemented by subclasses.
        """

    def info(self):
        """
        Method to return additional information about the observation. Can be overridden by subclasses.
        """
        return None
