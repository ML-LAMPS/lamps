import unittest
from types import SimpleNamespace

import numpy as np

from lamps.observers.log_loss_dynamics import LogLossDynamicsObserver


def make_env(order=("a", "b")):
    data = {
        "a": {"eval/log_loss": np.log([4.0, 2.0, 1.0, 0.5])},
        "b": {"eval/log_loss": np.log([1.0, 2.0, 4.0, 8.0])},
    }
    return SimpleNamespace(
        repository=SimpleNamespace(data=data),
        models=list(order),
        num_models=len(order),
        epoch_counts=np.zeros(len(order), dtype=np.int32),
    )


class LogLossDynamicsObserverTest(unittest.TestCase):
    def test_relative_improvement_and_recent_slope(self):
        env = make_env()
        observer = LogLossDynamicsObserver(env)
        np.testing.assert_array_equal(observer.observe(), np.zeros((2, 2)))

        env.epoch_counts[:] = [2, 1]
        observation = observer.observe()
        expected = np.asarray(
            [
                [np.tanh(np.log(4.0)), np.tanh(np.log(4.0) / 2)],
                [np.tanh(-np.log(2.0)), np.tanh(-np.log(2.0))],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observation, expected)
        self.assertTrue(observer.to_space().contains(observation))

    def test_recent_slope_uses_only_configured_window(self):
        env = make_env(("a",))
        observer = LogLossDynamicsObserver(env)
        env.epoch_counts[0] = 3
        observation = observer.observe()[0]
        self.assertAlmostEqual(
            float(observation[1]), float(np.tanh(np.log(8.0) / 3)), places=6
        )

    def test_features_are_permutation_equivariant(self):
        env = make_env()
        env.epoch_counts[:] = [2, 1]
        original = LogLossDynamicsObserver(env).observe().copy()

        permuted_env = make_env(("b", "a"))
        permuted_env.epoch_counts[:] = [1, 2]
        permuted = LogLossDynamicsObserver(permuted_env).observe()
        np.testing.assert_allclose(permuted, original[[1, 0]])


if __name__ == "__main__":
    unittest.main()
