"""Precompute strictly independent per-model log-loss GP posteriors."""

import argparse
import json
from pathlib import Path

import numpy as np

from lamps.independent_loss import (
    CACHE_KIND,
    DEFAULT_CACHE_DIR,
    IndependentLossCurveGP,
    cache_path,
    precompute_curve,
)
from lamps.posterior_cache import LogLossPosteriorCache
from lamps.repository import Repository


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True)
parser.add_argument("--datasets", required=True, help="Comma-separated datasets, or 'all'.")
parser.add_argument("--length-scale", type=float, default=50.0)
parser.add_argument("--signal-std", type=float, default=1.0)
parser.add_argument("--noise-std", type=float, default=0.05)
parser.add_argument("--jitter", type=float, default=1e-8)


def parse_datasets(value: str, experiment: str) -> list[str]:
    if value == "all":
        return Repository.list_datasets(experiment)
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def diagnostics(repository, means, stds, available_epochs) -> dict:
    actual = []
    predicted = []
    persistence = []
    uncertainty = []
    for model_index, model in enumerate(repository.models):
        curve = np.asarray(repository.data[model]["eval/log_loss"])
        available = int(available_epochs[model_index])
        actual.extend(curve[1 : available + 1])
        predicted.extend(means[model_index, :available])
        persistence.extend(curve[:available])
        uncertainty.extend(stds[model_index, :available])

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    persistence = np.asarray(persistence)
    uncertainty = np.asarray(uncertainty)
    error = predicted - actual
    return {
        "num_predictions": int(len(actual)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "persistence_rmse": float(
            np.sqrt(np.mean(np.square(persistence - actual)))
        ),
        "persistence_mae": float(np.mean(np.abs(persistence - actual))),
        "coverage_95": float(np.mean(np.abs(error) <= 1.96 * uncertainty)),
        "mean_posterior_std": float(np.mean(uncertainty)),
    }


def precompute_dataset(
    experiment: str,
    dataset: str,
    estimator: IndependentLossCurveGP,
) -> Path:
    repository = Repository(experiment, dataset, ["eval/log_loss"])
    available_epochs = repository.get_available_epochs().astype(np.int32)
    width = int(available_epochs.max()) + 1
    means = np.empty((repository.num_models, width), dtype=np.float32)
    stds = np.empty_like(means)

    for model_index, model in enumerate(repository.models):
        curve = np.asarray(repository.data[model]["eval/log_loss"], dtype=np.float32)
        available = int(available_epochs[model_index])
        model_means, model_stds = precompute_curve(curve, available, estimator)
        means[model_index, : available + 1] = model_means
        stds[model_index, : available + 1] = model_stds
        means[model_index, available + 1 :] = model_means[-1]
        stds[model_index, available + 1 :] = 0.0

    report = diagnostics(repository, means, stds, available_epochs)
    metadata = {
        "kind": CACHE_KIND,
        "metric": "eval/log_loss",
        "conditioning": "own_causal_prefix_only",
        "uses_other_models": False,
        "uses_other_datasets": False,
        "length_scale": estimator.length_scale,
        "signal_std": estimator.signal_std,
        "noise_std": estimator.noise_std,
        "jitter": estimator.jitter,
        "diagnostics": report,
    }
    output_path = cache_path(DEFAULT_CACHE_DIR, experiment, dataset)
    LogLossPosteriorCache(
        experiment=experiment,
        dataset=dataset,
        models=np.asarray(repository.models),
        available_epochs=available_epochs,
        mean=means,
        std=stds,
        metadata_json=json.dumps(metadata, sort_keys=True),
    ).save(output_path)
    print(f"wrote {output_path}")
    print(f"one-step diagnostics: {json.dumps(report, sort_keys=True)}")
    return output_path


def main():
    args = parser.parse_args()
    estimator = IndependentLossCurveGP(
        length_scale=args.length_scale,
        signal_std=args.signal_std,
        noise_std=args.noise_std,
        jitter=args.jitter,
    )
    for dataset in parse_datasets(args.datasets, args.experiment):
        precompute_dataset(args.experiment, dataset, estimator)


if __name__ == "__main__":
    main()
