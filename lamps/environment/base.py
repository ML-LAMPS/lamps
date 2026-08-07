from typing import Any

import numpy as np
import gymnasium as gym

from pymoo.indicators.hv import HV

from lamps.repository import Repository
from lamps import settings
from lamps import utils

OPTIMAL_REWARD = 3000.0


class DatasetEnv(gym.Env):
    """
    Simulates the progressive training of a fixed pool of models on a dataset,
    exposing an RL-friendly interface (obs/action spaces, reward, termination)
    built on top of tensorboard logs stored in the repository.
    """

    @property
    def models(self):
        return self.repository.models

    @property
    def num_models(self):
        return len(self.models)

    @property
    def objectives(self):
        return list(self.objectives_data.keys())

    @property
    def num_completed_models(self):
        return self.action_masks().count(False)

    @property
    def optimal_runtime(self):
        return self.repository.get_total_time(only_pareto=True)

    @property
    def max_runtime(self):
        return self.repository.get_total_time(only_pareto=False)

    @property
    def epoch_counts(self) -> np.ndarray:
        assert "epochs" in self.observers, "EpochObserver is not registered."
        return self.observers["epochs"].epoch_counts

    @property
    def available_epochs(self) -> np.ndarray:
        assert "epochs" in self.observers, "EpochObserver is not registered."
        return self.observers["epochs"].available_epochs

    @property
    def valid_mask(self) -> np.ndarray:
        assert "action_mask" in self.observers, "ActionMaskObserver is not registered."
        return self.observers["action_mask"].valid_mask

    def __init__(
        self,
        experiment: str,
        dataset: str,
        objectives: dict,
        ref_point: list | None = None,
        observers: list[str] | None = settings.OBSERVERS,
    ):
        self.repository = Repository(experiment, dataset, list(objectives.keys()))

        self.dataset = dataset
        self.objectives_data = objectives
        self.observers = self._setup_observers(observers)

        if ref_point is not None:
            self.ref_point = ref_point
        else:
            self.ref_point = self._compute_ref_point()

        assert len(objectives.keys()) == len(
            self.ref_point
        ), "Number of objectives and reference point dimensions must match."

        self.optimal_hv = self.repository.get_optimal_hv(self.ref_point)
        self.pareto_models = self.repository.get_pareto_models()
        self.pareto_models_idx = [
            self.models.index(model) for model in self.pareto_models
        ]

        num_models = self.repository.num_models

        obs_space = {
            # "action_mask": gym.spaces.Box(
            #     low=0,
            #     high=1,
            #     shape=(num_models,),
            #     dtype=np.float32,
            # ),
        }

        for key, value in objectives.items():
            obs_space[key] = gym.spaces.Box(
                low=value["min_value"],
                high=value["max_value"],
                shape=(num_models,),
                dtype=np.float32,
            )

        for observer in self.observers.values():
            obs_space[observer.NAME] = observer.to_space()

        self.observation_space = gym.spaces.Dict(obs_space)
        self.action_space = gym.spaces.Discrete(num_models)

        # self.available_epochs = np.array(
        #     [self.repository.get_num_available_epochs(model) for model in self.models],
        #     dtype=np.int32,
        # )

        self.objective_arrays = {
            objective: [
                np.asarray(self.repository.data[model][objective], dtype=np.float32)
                for model in self.models
            ]
            for objective in self.objectives
        }

    def _setup_observers(self, observers: list[str]):

        instances = {}

        for obs in observers:
            obs_cls = utils.load_class(obs)
            instances[obs_cls.NAME] = obs_cls(self)

            print(f"Registered observer: {obs_cls.NAME} ({obs_cls.__name__})")
        return instances

    def _compute_ref_point(self, gap: float = 1.1):
        """
        For each objective, compute a reference point that is slightly worse
        than the worst observed value across all models and epochs.
        """
        ref_point = []

        for metric in self.objectives:

            if metric in ("eval/neg_accuracy", "eval/neg_f1_macro"):
                worst_value = 0.0
            else:
                worst_value = -np.inf

                for model in self.repository.models:
                    max_value = np.max(self.repository.data[model][metric])
                    worst_value = max(worst_value, max_value)

            ref_point.append(float(worst_value * gap))

        return ref_point

    def search_time(self):
        return self.observers["runtime"].search_time()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        for observer in self.observers.values():
            observer.reset()

        # valid_mask = self._compute_valid_mask()

        observation = self.get_obs(self.valid_mask)
        info = self.get_info()

        return observation, info

    def _compute_valid_mask(self) -> np.ndarray:
        return self.observers["action_mask"].compute_valid_mask()

    def get_obs(self, valid_mask: np.ndarray | None = None):
        if valid_mask is None:
            valid_mask = self._compute_valid_mask()

        obs = {
            "action_mask": valid_mask.astype(np.float32),
        }

        for observer in self.observers.values():
            obs[observer.NAME] = observer.observe()

        # Add objective metrics to observation
        for objective in self.objectives:
            objective_values = np.fromiter(
                (
                    self.objective_arrays[objective][i][epoch]
                    for i, epoch in enumerate(self.epoch_counts)
                ),
                dtype=np.float32,
                count=self.num_models,
            )
            obs[objective] = objective_values

        return obs

    def get_info(self):

        info = {}

        for observer in self.observers.values():
            _info = observer.info()

            if _info:
                info.update(_info)

        return info

    def num_remaining_models(self):
        return int(np.count_nonzero(self._compute_valid_mask()))

    def valid_actions(self):
        return np.flatnonzero(self._compute_valid_mask()).tolist()

    def invalid_actions(self):
        return np.flatnonzero(~self._compute_valid_mask()).tolist()

    def action_masks(self) -> list[bool]:
        return self._compute_valid_mask().tolist()

    def step(self, action):
        """
        Execute one training epoch for the selected model (action).
        """
        model_idx = int(action)
        model = self.models[model_idx]
        valid_mask_before = self.valid_mask

        assert valid_mask_before[
            model_idx
        ], f"Model {model} (action {model_idx}) is fully trained."

        for observer in self.observers.values():
            observer.step(action)

        valid_mask_after = self.valid_mask

        terminated = self.is_terminated(valid_mask_after)
        reward = self.reward(terminated, valid_mask_after)
        truncated = False
        observation = self.get_obs(valid_mask_after)
        info = self.get_info()

        return observation, reward, terminated, truncated, info

    def reward(
        self, terminated: bool | None = None, valid_mask: np.ndarray | None = None
    ):

        if terminated is None:
            terminated = self.is_terminated(valid_mask)

        if not terminated:
            return 0.0

        num_pareto_models = len(self.pareto_models)
        if valid_mask is None:
            valid_mask = self.valid_mask
        num_completed_models = self.num_models - int(np.count_nonzero(valid_mask))

        _reward = (self.num_models - num_completed_models) / self.search_time()
        _optimal = (self.num_models - num_pareto_models) / self.optimal_runtime

        return OPTIMAL_REWARD * (_reward / _optimal)

    def is_terminated(self, valid_mask: np.ndarray | None = None):
        # return self.hypervolume() >= self.optimal_hv

        # FIXME: This condition is not ideal for slowly converging scenarios. Checking
        # the hypervolume is a better approach, but it requires a different reward.
        if valid_mask is None:
            valid_mask = self.valid_mask

        return bool(np.all(~valid_mask[self.pareto_models_idx]))

    def hypervolume(self, only_finished: bool = False):

        epoch_counts = self.epoch_counts.copy()

        if only_finished:
            for i in self.valid_actions():
                epoch_counts[i] = 0

        datapoints = self.repository.datapoints(epoch_counts, preserve_best=True)
        hv = HV(ref_point=self.ref_point)
        hv_value = hv.do(datapoints)

        if hv_value is None:
            raise ValueError(
                "Hypervolume calculation failed. Check if the reference point is set correctly."
            )

        return hv_value

    def render(self, mode="human"):
        pass
