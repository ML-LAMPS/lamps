import unittest
from types import SimpleNamespace

import numpy as np

from lamps.observers.pareto_dominance import ParetoDominanceObserver


def make_env(order=("a", "b", "c")):
    data = {
        "a": {"size": [1.0, 1.0], "loss": [3.0, 3.0]},
        "b": {"size": [2.0, 2.0], "loss": [2.0, 2.0]},
        "c": {"size": [3.0, 3.0], "loss": [4.0, 1.0]},
    }
    return SimpleNamespace(
        repository=SimpleNamespace(data=data),
        models=list(order),
        num_models=len(order),
        objectives=["size", "loss"],
        epoch_counts=np.zeros(len(order), dtype=np.int32),
    )


class ParetoDominanceObserverTest(unittest.TestCase):
    def test_dominance_features_update_causally(self):
        env = make_env()
        observer = ParetoDominanceObserver(env)

        np.testing.assert_allclose(
            observer.observe(),
            [
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0],
                [1.0, 0.0, 0.0],
            ],
        )

        # Revealing c's second point makes all three models mutually
        # non-dominating. No unrevealed value was used before this update.
        env.epoch_counts[2] = 1
        np.testing.assert_allclose(
            observer.observe(),
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
        )
        self.assertTrue(observer.to_space().contains(observer.observe()))

    def test_features_are_permutation_equivariant(self):
        original = ParetoDominanceObserver(make_env()).observe().copy()
        permutation = np.asarray([2, 0, 1])
        permuted = ParetoDominanceObserver(make_env(("c", "a", "b"))).observe()
        np.testing.assert_allclose(permuted, original[permutation])

    def test_single_model_is_on_front(self):
        env = make_env(("a",))
        observer = ParetoDominanceObserver(env)
        np.testing.assert_allclose(observer.observe(), [[0.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
