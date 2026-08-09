"""Causal, bounded learning-curve dynamics for eval/log_loss."""

import gymnasium as gym
import numpy as np

from .base import BaseObserver


class LogLossDynamicsObserver(BaseObserver):
    """Expose relative improvement and recent slope for each model.

    Since ``eval/log_loss`` is logarithmic, subtracting two observations is a
    log ratio of the corresponding losses. Positive values mean improvement.
    ``tanh`` keeps both features on a stable, bounded scale.
    """

    NAME = "eval/log_loss_dynamics"
    FEATURES = ("relative_improvement", "recent_slope")
    SLOPE_WINDOW = 3

    def __init__(self, env):
        super().__init__(env)
        self._curves = tuple(
            np.asarray(
                self.repository.data[model]["eval/log_loss"], dtype=np.float32
            )
            for model in self.models
        )
        if any(curve.ndim != 1 or curve.size == 0 for curve in self._curves):
            raise ValueError("Log-loss curves must be non-empty and one-dimensional.")
        if any(not np.all(np.isfinite(curve)) for curve in self._curves):
            raise ValueError("Log-loss curves must contain only finite values.")

        self._epoch_counts = np.empty(self.num_models, dtype=np.int32)
        self._observation = np.empty(
            (self.num_models, len(self.FEATURES)), dtype=np.float32
        )
        self.reset()

    def to_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_models, len(self.FEATURES)),
            dtype=np.float32,
        )

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

    def _update_model(self, model_index: int, epoch: int) -> None:
        curve = self._curves[model_index]
        current_loss = curve[epoch]
        relative_improvement = curve[0] - current_loss

        window_start = max(0, epoch - self.SLOPE_WINDOW)
        elapsed_epochs = epoch - window_start
        if elapsed_epochs:
            recent_slope = (curve[window_start] - current_loss) / elapsed_epochs
        else:
            recent_slope = 0.0

        self._observation[model_index] = np.tanh(
            [relative_improvement, recent_slope]
        ).astype(np.float32)

    def reset(self):
        counts = self._validated_epoch_counts()
        for model_index, epoch in enumerate(counts):
            self._update_model(model_index, int(epoch))
        self._epoch_counts[:] = counts

    def observe(self):
        counts = self._validated_epoch_counts()
        changed = np.flatnonzero(counts != self._epoch_counts)
        for model_index in changed:
            self._update_model(model_index, int(counts[model_index]))
        if changed.size:
            self._epoch_counts[:] = counts
        return self._observation
