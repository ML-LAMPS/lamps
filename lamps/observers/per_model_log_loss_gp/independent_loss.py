"""Strictly independent Bayesian next-epoch learning-curve estimation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .posterior_cache import posterior_cache_path


DEFAULT_CACHE_DIR = Path(".cache/per_model_log_loss_gp")
CACHE_KIND = "independent_matern32_gp"


def cache_path(cache_dir: str | Path, experiment: str, dataset: str) -> Path:
    return posterior_cache_path(cache_dir, experiment, dataset)


@dataclass(frozen=True)
class IndependentLossCurveGP:
    """Fixed-prior Matérn-3/2 GP fit to one model's revealed prefix only."""

    length_scale: float = 50.0
    signal_std: float = 1.0
    noise_std: float = 0.05
    jitter: float = 1e-8

    def __post_init__(self):
        if self.length_scale <= 0 or self.signal_std <= 0:
            raise ValueError("GP length_scale and signal_std must be positive.")
        if self.noise_std < 0 or self.jitter <= 0:
            raise ValueError("GP noise_std must be non-negative and jitter positive.")

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = np.abs(left[:, None] - right[None, :])
        scaled = np.sqrt(3.0) * distance / self.length_scale
        return self.signal_std**2 * (1.0 + scaled) * np.exp(-scaled)

    def predict_next(self, prefix: np.ndarray) -> tuple[float, float]:
        values = np.asarray(prefix, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("An observed one-dimensional curve prefix is required.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Curve prefix contains non-finite values.")

        epochs = np.arange(len(values), dtype=np.float64)
        query = np.asarray([float(len(values))])
        covariance = self._kernel(epochs, epochs)
        covariance.flat[:: len(values) + 1] += self.noise_std**2 + self.jitter
        query_covariance = self._kernel(epochs, query)[:, 0]

        # A constant y_0 mean makes the first prediction a persistence forecast;
        # subsequent observations update it through the GP posterior.
        centered = values - values[0]
        cholesky = np.linalg.cholesky(covariance)
        weights = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, centered))
        mean = float(values[0] + query_covariance @ weights)

        projected = np.linalg.solve(cholesky, query_covariance)
        latent_variance = max(self.signal_std**2 - float(projected @ projected), 0.0)
        predictive_variance = latent_variance + self.noise_std**2
        return mean, float(np.sqrt(max(predictive_variance, self.jitter)))


def precompute_curve(
    curve: np.ndarray, available_epoch: int, estimator: IndependentLossCurveGP
) -> tuple[np.ndarray, np.ndarray]:
    """Predict each next epoch using only the prefix available at that index."""
    values = np.asarray(curve, dtype=np.float32)
    if available_epoch < 0 or available_epoch >= len(values):
        raise ValueError("available_epoch is outside the learning curve.")
    means = np.empty(available_epoch + 1, dtype=np.float32)
    stds = np.empty_like(means)
    for current_epoch in range(available_epoch):
        mean, std = estimator.predict_next(values[: current_epoch + 1])
        means[current_epoch] = np.clip(mean, -15.0, 15.0)
        stds[current_epoch] = np.clip(std, 0.0, 15.0)
    means[available_epoch] = np.clip(values[available_epoch], -15.0, 15.0)
    stds[available_epoch] = 0.0
    return means, stds
