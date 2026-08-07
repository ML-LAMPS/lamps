from lamps.environment.base import DatasetEnv


class TrainingDatasetEnv(DatasetEnv):
    """Used for RL training: no history tracking or rendering overhead."""
