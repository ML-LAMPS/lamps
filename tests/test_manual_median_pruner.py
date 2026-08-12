import unittest

from lamps.hpo import ManualMedianPruner


class ManualMedianPrunerTest(unittest.TestCase):
    def test_no_pruning_during_startup(self):
        pruner = ManualMedianPruner(n_warmup_evals=0, n_startup_trials=3)

        self.assertFalse(pruner.should_prune(eval_index=0, value=-1000.0))

    def test_no_pruning_during_warmup_evals(self):
        pruner = ManualMedianPruner(n_warmup_evals=2, n_startup_trials=0)
        pruner.record_completed_trial([10.0, 10.0, 10.0])

        self.assertFalse(pruner.should_prune(eval_index=1, value=-1000.0))
        self.assertTrue(pruner.should_prune(eval_index=2, value=-1000.0))

    def test_prunes_below_historical_median(self):
        pruner = ManualMedianPruner(n_warmup_evals=0, n_startup_trials=0)
        pruner.record_completed_trial([0.0, 5.0])
        pruner.record_completed_trial([0.0, 15.0])

        self.assertTrue(pruner.should_prune(eval_index=1, value=9.0))
        self.assertFalse(pruner.should_prune(eval_index=1, value=11.0))

    def test_pruned_trials_do_not_pollute_history(self):
        pruner = ManualMedianPruner(n_warmup_evals=0, n_startup_trials=0)
        pruner.record_completed_trial([10.0])

        # A trial that gets pruned is never recorded, so the bar doesn't drop.
        self.assertTrue(pruner.should_prune(eval_index=0, value=1.0))
        self.assertEqual(pruner.history[0], [10.0])


if __name__ == "__main__":
    unittest.main()
