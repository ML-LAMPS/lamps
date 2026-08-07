from abc import ABC, abstractmethod

from gymnasium.spaces import Space


class BaseObservation(ABC):
    """
    Base class for observations in the LAMPS environment.
    """

    NAME: str = None

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def reset(self):
        """
        Method to reset the observation. Should be implemented by subclasses.
        """

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

    def step(self, action):
        """
        Method to update the observation based on the action taken. Should be implemented by subclasses.
        """

    def info(self):
        """
        Method to return additional information about the observation. Can be overridden by subclasses.
        """
        return None
