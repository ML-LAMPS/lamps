import unittest

import numpy as np

from lamps.observers.per_model_log_loss_gp.independent_loss import (
    IndependentLossCurveGP,
    precompute_curve,
)


class IndependentLossCurveTest(unittest.TestCase):
    def test_future_values_cannot_change_earlier_predictions(self):
        estimator = IndependentLossCurveGP()
        original = np.asarray([2.0, 1.8, 1.7, 1.6, 1.5], dtype=np.float32)
        poisoned = original.copy()
        poisoned[3:] = [100.0, -100.0]

        original_mean, original_std = precompute_curve(original, 4, estimator)
        poisoned_mean, poisoned_std = precompute_curve(poisoned, 4, estimator)

        # Predictions at current epochs 0, 1, and 2 can only use values 0..2.
        np.testing.assert_array_equal(original_mean[:3], poisoned_mean[:3])
        np.testing.assert_array_equal(original_std[:3], poisoned_std[:3])

    def test_first_prediction_is_persistence_with_positive_uncertainty(self):
        estimator = IndependentLossCurveGP()
        mean, std = estimator.predict_next(np.asarray([2.5]))
        self.assertAlmostEqual(mean, 2.5)
        self.assertGreater(std, 0.0)

    def test_hyperparameters_are_validated(self):
        with self.assertRaisesRegex(ValueError, "length_scale"):
            IndependentLossCurveGP(length_scale=0.0)


if __name__ == "__main__":
    unittest.main()
