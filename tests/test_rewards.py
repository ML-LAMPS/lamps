import unittest

import numpy as np

from lamps.rewards import SparseReward, PotentialShapedReward
from lamps.rewards.sparse import OPTIMAL_REWARD


class FakeEnv:
    def __init__(self):
        self.num_models = 3
        self.pareto_models = ["a", "b"]
        self.pareto_models_idx = [0, 1]
        self.optimal_runtime = 5.0
        self.valid_mask = np.array([True, True, True])
        self.epoch_counts = np.zeros(3, dtype=np.intp)
        self.available_epochs = np.array([10, 10, 10])

    def search_time(self):
        return 10.0


class SparseRewardTest(unittest.TestCase):
    def test_zero_when_not_terminated(self):
        reward_fn = SparseReward(FakeEnv())
        self.assertEqual(reward_fn.compute(False), 0.0)

    def test_positive_when_terminated(self):
        env = FakeEnv()
        reward_fn = SparseReward(env)
        self.assertGreater(reward_fn.compute(True, env.valid_mask), 0.0)


class PotentialShapedRewardTest(unittest.TestCase):
    def test_potential_starts_at_zero_and_scales_with_pareto_progress(self):
        env = FakeEnv()
        reward_fn = PotentialShapedReward(env, shaping_scale=0.02)

        self.assertEqual(reward_fn.potential(), 0.0)

        env.epoch_counts[0] = 5
        expected = 0.02 * OPTIMAL_REWARD * 5 / 20
        self.assertAlmostEqual(reward_fn.potential(), expected)

    def test_non_pareto_progress_does_not_move_potential(self):
        env = FakeEnv()
        reward_fn = PotentialShapedReward(env, shaping_scale=0.02)

        env.epoch_counts[2] = 7  # model "c" is not Pareto-optimal
        self.assertEqual(reward_fn.potential(), 0.0)

    def test_non_terminal_reward_equals_shaping_term_only(self):
        env = FakeEnv()
        reward_fn = PotentialShapedReward(env, gamma=0.9, shaping_scale=0.02)

        reward_fn.before_step(action=0)
        env.epoch_counts[0] = 1

        value = reward_fn.compute(False, env.valid_mask)
        info = reward_fn.info()

        self.assertEqual(info["reward/unshaped"], 0.0)
        self.assertAlmostEqual(value, info["reward/shaping"])

    def test_terminal_reward_equals_sparse_component_minus_last_potential(self):
        env = FakeEnv()
        reward_fn = PotentialShapedReward(env, gamma=0.9, shaping_scale=0.02)

        reward_fn.before_step(action=0)
        value = reward_fn.compute(True, env.valid_mask)
        info = reward_fn.info()

        self.assertAlmostEqual(info["reward/shaping"], -reward_fn._potential_before)
        self.assertAlmostEqual(value, info["reward/unshaped"] + info["reward/shaping"])


if __name__ == "__main__":
    unittest.main()
