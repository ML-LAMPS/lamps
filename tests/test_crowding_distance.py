import unittest

import numpy as np

from lamps.observers.crowding_distance import (
    CrowdingDistanceObserver,
    CROWDING_DISTANCE_CAP,
)


class FakeRepository:
    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)

    def datapoints(self, epoch_counts):
        return self._points


class FakeEnv:
    def __init__(self, points):
        self.repository = FakeRepository(points)
        self.num_models = len(points)
        self.epoch_counts = np.zeros(self.num_models, dtype=np.intp)


class CrowdingDistanceObserverTest(unittest.TestCase):
    def test_boundary_models_get_capped_distance(self):
        env = FakeEnv([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]])
        observer = CrowdingDistanceObserver(env)

        observation = observer.observe()

        self.assertEqual(observation[0], CROWDING_DISTANCE_CAP)
        self.assertEqual(observation[2], CROWDING_DISTANCE_CAP)
        self.assertLess(observation[1], CROWDING_DISTANCE_CAP)
        self.assertGreaterEqual(observation[1], 0.0)

    def test_two_models_are_both_boundary(self):
        env = FakeEnv([[1.0, 1.0], [2.0, 2.0]])
        observer = CrowdingDistanceObserver(env)

        observation = observer.observe()

        np.testing.assert_array_equal(
            observation, [CROWDING_DISTANCE_CAP, CROWDING_DISTANCE_CAP]
        )

    def test_observation_is_contained_in_declared_space(self):
        env = FakeEnv([[0.0, 10.0], [3.0, 7.0], [5.0, 5.0], [7.0, 3.0], [10.0, 0.0]])
        observer = CrowdingDistanceObserver(env)

        observation = observer.observe()

        self.assertTrue(observer.to_space().contains(observation))


if __name__ == "__main__":
    unittest.main()
