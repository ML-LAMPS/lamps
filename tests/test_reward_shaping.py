import unittest

from lamps.rewards.potential_shaped import potential_shaping


class PotentialShapingTest(unittest.TestCase):
    def _discounted_sum(self, potentials, terminated_at, gamma):
        total = 0.0
        discount = 1.0
        for t in range(len(potentials) - 1):
            terminated = t + 1 == terminated_at
            total += discount * potential_shaping(
                potentials[t], potentials[t + 1], terminated, gamma
            )
            discount *= gamma
        return total

    def test_discounted_shaping_sum_depends_only_on_initial_potential(self):
        gamma = 0.9
        phi0 = 5.0

        short_path = [phi0, 3.0, 9.0]
        long_path = [phi0, 1.0, 8.0, 2.0, 6.0]

        short_total = self._discounted_sum(short_path, terminated_at=2, gamma=gamma)
        long_total = self._discounted_sum(long_path, terminated_at=4, gamma=gamma)

        self.assertAlmostEqual(short_total, -phi0)
        self.assertAlmostEqual(long_total, -phi0)

    def test_non_terminal_step_uses_raw_next_potential(self):
        term = potential_shaping(2.0, 7.0, terminated=False, gamma=0.9)
        self.assertAlmostEqual(term, 0.9 * 7.0 - 2.0)

    def test_terminal_step_zeroes_next_potential(self):
        term = potential_shaping(2.0, 7.0, terminated=True, gamma=0.9)
        self.assertAlmostEqual(term, 0.9 * 0.0 - 2.0)


if __name__ == "__main__":
    unittest.main()
