from typing import Any

import numpy as np
import gymnasium as gym

from pymoo.indicators.hv import HV

from lamps.repository import Repository

NUM_MAX_EPOCHS = 50
MAX_RUNTIME = 72 * 60 * 60.0  # 72 hours in seconds
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

    def __init__(
        self,
        experiment: str,
        dataset: str,
        objectives: dict,
        ref_point: list | None = None,
        skip_models: list = [],
    ):
        self.repository = Repository(experiment, dataset, list(objectives.keys()))

        self.dataset = dataset
        self.objectives_data = objectives
        self.skip_models = skip_models
        self.skip_models_indices = [
            self.models.index(model) for model in self.skip_models
        ]

        if ref_point is not None:
            self.ref_point = ref_point
        else:
            self.ref_point = self._compute_ref_point()

        assert len(objectives.keys()) == len(
            self.ref_point
        ), "Number of objectives and reference point dimensions must match."

        self.optimal_hv = self.repository.get_optimal_hv(self.ref_point)
        self.pareto_models = self.repository.get_pareto_models(
            exclude_models=self.skip_models
        )
        self.pareto_models_idx = [
            self.models.index(model) for model in self.pareto_models
        ]

        num_models = self.repository.num_models

        obs_space = {
            "epochs": gym.spaces.Box(
                low=0,
                high=NUM_MAX_EPOCHS,
                shape=(num_models,),
                dtype=np.float32,
            ),
            "runtime": gym.spaces.Box(
                low=0,
                high=MAX_RUNTIME,
                shape=(num_models,),
                dtype=np.float32,
            ),
            "action_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(num_models,),
                dtype=np.float32,
            ),
        }

        for key, value in objectives.items():
            obs_space[key] = gym.spaces.Box(
                low=value["min_value"],
                high=value["max_value"],
                shape=(num_models,),
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(obs_space)
        self.action_space = gym.spaces.Discrete(num_models)

        self.epoch_counts = np.zeros(num_models, dtype=np.int32)

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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        self.epoch_counts = np.zeros(self.num_models, dtype=np.int32)

        observation = self.get_obs()
        info = self.get_info()

        return observation, info

    def get_obs(self):
        current_runtime = np.array(
            [
                self.repository.get_elapsed_time(model, self.epoch_counts[i])
                for i, model in enumerate(self.repository.models)
            ]
        )

        obs = {
            "epochs": self.epoch_counts,
            "runtime": current_runtime,
            "action_mask": np.array(self.action_masks(), dtype=np.float32),
        }

        # Add objective metrics to observation
        for objective in self.objectives:
            obs[objective] = [
                self.repository.get_metric(model, objective, self.epoch_counts[i])
                for i, model in enumerate(self.repository.models)
            ]

        return obs

    def get_info(self):
        return {
            "epoch_counts": self.epoch_counts,
        }

    def num_remaining_models(self):
        return self.action_masks().count(True)

    def search_time(self):
        total_time = 0.0
        for i, model in enumerate(self.repository.models):
            total_time += self.repository.get_elapsed_time(model, self.epoch_counts[i])

        return total_time

    def valid_actions(self):
        epochs_remaining = (
            np.array(
                [
                    self.repository.get_num_available_epochs(model)
                    for model in self.repository.models
                ]
            )
            - self.epoch_counts
        )
        valid = np.where(epochs_remaining > 0)[0].tolist()
        if self.skip_models_indices:
            valid = [i for i in valid if i not in self.skip_models_indices]
        return valid

    def invalid_actions(self):
        return list(set(range(self.num_models)) - set(self.valid_actions()))

    def action_masks(self) -> list[bool]:
        mask = np.zeros(self.num_models, dtype=np.int8)
        mask[self.valid_actions()] = 1
        return mask.astype(bool).tolist()

    def step(self, action):
        """
        Execute one training epoch for the selected model (action).
        """
        model_idx = int(action)
        model = self.models[model_idx]

        assert (
            model_idx in self.valid_actions()
        ), f"Model {model} (action {model_idx}) is fully trained."

        # Update epoch count
        self.epoch_counts[model_idx] += 1

        reward = self.reward()
        terminated = self.is_terminated()
        truncated = False
        observation = self.get_obs()
        info = self.get_info()

        return observation, reward, terminated, truncated, info

    def reward(self):

        if not self.is_terminated():
            return 0.0

        num_pareto_models = len(self.pareto_models)

        _reward = (self.num_models - self.num_completed_models) / self.search_time()
        _optimal = (self.num_models - num_pareto_models) / self.optimal_runtime

        return OPTIMAL_REWARD * (_reward / _optimal)

    def is_terminated(self):
        # return self.hypervolume() >= self.optimal_hv

        # FIXME: This condition is not ideal for slowly converging scenarios. Checking
        # the hypervolume is a better approach, but it requires a different reward.
        return set(self.pareto_models_idx).issubset(self.invalid_actions())

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
