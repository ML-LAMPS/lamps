"""Causal, permutation-equivariant Pareto-dominance features."""

import gymnasium as gym
import numpy as np

from .base import BaseObserver


class ParetoDominanceObserver(BaseObserver):
    """Describe each model's current position relative to the model pool.

    All repository objectives are represented as minimization objectives. The
    observer exposes, in order, the fraction of other models that dominate a
    model, the fraction it dominates, and whether it is on the current front.
    Only values at the environment's current epoch counts are used.
    """

    NAME = "multi_objective/dominance"
    FEATURES = (
        "dominated_by_fraction",
        "dominates_fraction",
        "current_front",
    )

    def __init__(self, env):
        super().__init__(env)
        if not self.env.objectives:
            raise ValueError("Pareto dominance requires at least one objective.")

        self._curves = tuple(
            np.column_stack(
                [
                    self.repository.data[model][objective]
                    for objective in self.env.objectives
                ]
            ).astype(np.float64, copy=False)
            for model in self.models
        )
        if any(not np.all(np.isfinite(curve)) for curve in self._curves):
            raise ValueError("Pareto-dominance objective curves must be finite.")

        self._epoch_counts = np.empty(self.num_models, dtype=np.int32)
        self._current_values = np.empty(
            (self.num_models, len(self.env.objectives)), dtype=np.float64
        )
        self._dominance = np.zeros(
            (self.num_models, self.num_models), dtype=bool
        )
        self._observation = np.empty((self.num_models, 3), dtype=np.float32)
        self.reset()

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_models, len(self.FEATURES)),
            dtype=np.float32,
        )

    @staticmethod
    def _dominance_matrix(values: np.ndarray) -> np.ndarray:
        """Return matrix D where D[i, j] means model i dominates model j."""
        no_worse = np.all(values[:, None, :] <= values[None, :, :], axis=2)
        strictly_better = np.any(values[:, None, :] < values[None, :, :], axis=2)
        dominance = no_worse & strictly_better
        np.fill_diagonal(dominance, False)
        return dominance

    def _validated_epoch_counts(self) -> np.ndarray:
        counts = np.asarray(self.env.epoch_counts, dtype=np.int32)
        if counts.shape != (self.num_models,):
            raise ValueError("Epoch counts must contain one value per model.")
        for model_index, epoch in enumerate(counts):
            if epoch < 0 or epoch >= len(self._curves[model_index]):
                raise ValueError(
                    f"Epoch {epoch} is outside model {self.models[model_index]!r}."
                )
        return counts

    def _refresh_observation(self) -> None:
        denominator = max(self.num_models - 1, 1)
        dominated_by = self._dominance.sum(axis=0) / denominator
        dominates = self._dominance.sum(axis=1) / denominator
        self._observation[:, 0] = dominated_by
        self._observation[:, 1] = dominates
        self._observation[:, 2] = dominated_by == 0.0

    def reset(self):
        counts = self._validated_epoch_counts()
        for model_index, epoch in enumerate(counts):
            self._current_values[model_index] = self._curves[model_index][epoch]
        self._dominance[:] = self._dominance_matrix(self._current_values)
        self._epoch_counts[:] = counts
        self._refresh_observation()

    def _sync_changed_models(self) -> None:
        counts = self._validated_epoch_counts()
        changed = np.flatnonzero(counts != self._epoch_counts)
        if changed.size == 0:
            return

        for model_index in changed:
            self._current_values[model_index] = self._curves[model_index][
                counts[model_index]
            ]

        # A model update can change only its own row and column. Updating these
        # is O(changed_models * models * objectives), rather than recomputing
        # the full pairwise matrix after every environment step.
        for model_index in changed:
            value = self._current_values[model_index]
            row = np.all(value <= self._current_values, axis=1) & np.any(
                value < self._current_values, axis=1
            )
            column = np.all(self._current_values <= value, axis=1) & np.any(
                self._current_values < value, axis=1
            )
            self._dominance[model_index, :] = row
            self._dominance[:, model_index] = column
            self._dominance[model_index, model_index] = False

        self._epoch_counts[:] = counts
        self._refresh_observation()

    def observe(self):
        self._sync_changed_models()
        return self._observation
