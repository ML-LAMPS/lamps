import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from lamps.observers.per_model_log_loss_gp import PerModelLogLossGPObserver
from lamps.observers.per_model_log_loss_gp.independent_loss import CACHE_KIND, cache_path
from lamps.observers.per_model_log_loss_gp.posterior_cache import LogLossPosteriorCache


class PosteriorCacheTest(unittest.TestCase):
    def test_round_trip_and_observer_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            models = np.asarray(["model-b", "model-a"])
            available = np.asarray([1, 2], dtype=np.int32)
            mean = np.asarray([[2.0, 2.1, 2.1], [1.0, 1.1, 1.2]])
            std = np.asarray([[0.5, 0.0, 0.0], [0.4, 0.3, 0.0]])
            path = cache_path(directory, "experiment", "dataset/config[x-y]")
            LogLossPosteriorCache(
                experiment="experiment",
                dataset="dataset/config[x-y]",
                models=models,
                available_epochs=available,
                mean=mean,
                std=std,
                metadata_json=json.dumps({"kind": CACHE_KIND}),
            ).save(path)

            loaded = LogLossPosteriorCache.load(path)
            np.testing.assert_array_equal(loaded.models, models)

            env = SimpleNamespace(
                repository=SimpleNamespace(experiment="experiment"),
                dataset="dataset/config[x-y]",
                models=["model-a", "model-b"],
                num_models=2,
                available_epochs=np.asarray([2, 1], dtype=np.int32),
                epoch_counts=np.asarray([1, 0], dtype=np.int32),
            )
            observer = PerModelLogLossGPObserver(env, cache_dir=directory)
            observation = observer.observe()
            np.testing.assert_allclose(observation, [[1.1, 0.3], [2.0, 0.5]])
            self.assertTrue(observer.to_space().contains(observation))

    def test_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            LogLossPosteriorCache(
                experiment="experiment",
                dataset="dataset",
                models=np.asarray(["model"]),
                available_epochs=np.asarray([0]),
                mean=np.asarray([[np.nan]]),
                std=np.asarray([[0.0]]),
            )


if __name__ == "__main__":
    unittest.main()
