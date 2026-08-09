"""Serialization and validation for precomputed log-loss posteriors."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np


CACHE_SCHEMA_VERSION = 1
METRIC = "eval/log_loss"


def posterior_cache_path(
    cache_dir: str | Path, experiment: str, dataset: str
) -> Path:
    """Return the unambiguous cache filename for an experiment/dataset pair."""
    experiment_component = quote(experiment, safe="-_.")
    dataset_component = quote(dataset, safe="-_.")
    return Path(cache_dir) / experiment_component / f"{dataset_component}.npz"


@dataclass(frozen=True)
class LogLossPosteriorCache:
    """Next-epoch posterior moments indexed by model and current epoch."""

    experiment: str
    dataset: str
    models: np.ndarray
    available_epochs: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    metadata_json: str = "{}"

    def __post_init__(self):
        models = np.asarray(self.models, dtype=str)
        available_epochs = np.asarray(self.available_epochs, dtype=np.int32)
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)

        if models.ndim != 1:
            raise ValueError("Posterior cache models must be one-dimensional.")
        if len(set(models.tolist())) != len(models):
            raise ValueError("Posterior cache contains duplicate model names.")
        if available_epochs.shape != (len(models),):
            raise ValueError("available_epochs must have one entry per model.")
        if mean.shape != std.shape or mean.ndim != 2:
            raise ValueError("Posterior mean/std must be equally shaped matrices.")
        if mean.shape[0] != len(models):
            raise ValueError("Posterior mean/std must have one row per model.")
        if np.any(available_epochs < 0) or np.any(available_epochs >= mean.shape[1]):
            raise ValueError("Available epoch index is outside the posterior cache.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("Posterior cache contains non-finite values.")
        if np.any(std < 0):
            raise ValueError("Posterior standard deviations must be non-negative.")

        object.__setattr__(self, "models", models)
        object.__setattr__(self, "available_epochs", available_epochs)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int32),
            metric=np.asarray(METRIC),
            experiment=np.asarray(self.experiment),
            dataset=np.asarray(self.dataset),
            models=self.models,
            available_epochs=self.available_epochs,
            mean=self.mean,
            std=self.std,
            metadata_json=np.asarray(self.metadata_json),
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "LogLossPosteriorCache":
        input_path = Path(path)
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Log-loss posterior cache not found: {input_path}. "
                "Run the corresponding precomputation command first."
            )

        with np.load(input_path, allow_pickle=False) as data:
            required = {
                "schema_version",
                "metric",
                "experiment",
                "dataset",
                "models",
                "available_epochs",
                "mean",
                "std",
                "metadata_json",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(
                    f"Posterior cache {input_path} is missing keys: {sorted(missing)}"
                )

            schema_version = int(np.asarray(data["schema_version"]).item())
            metric = str(np.asarray(data["metric"]).item())
            if schema_version != CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported posterior cache schema {schema_version}; "
                    f"expected {CACHE_SCHEMA_VERSION}."
                )
            if metric != METRIC:
                raise ValueError(
                    f"Posterior cache metric is {metric!r}; expected {METRIC!r}."
                )

            return cls(
                experiment=str(np.asarray(data["experiment"]).item()),
                dataset=str(np.asarray(data["dataset"]).item()),
                models=np.asarray(data["models"]).copy(),
                available_epochs=np.asarray(data["available_epochs"]).copy(),
                mean=np.asarray(data["mean"]).copy(),
                std=np.asarray(data["std"]).copy(),
                metadata_json=str(np.asarray(data["metadata_json"]).item()),
            )
