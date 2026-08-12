import numpy as np


class ManualMedianPruner:
    """
    Median-pruner replacement for Optuna multi-objective studies.

    Optuna's Trial.report()/should_prune() raise NotImplementedError when a
    study has multiple directions, so there is no built-in way to prune
    mid-training in a multi-objective study. This reimplements the same idea
    by hand: a trial is pruned if, at a given evaluation index, its value is
    below the chosen percentile of same-index values from trials that
    already ran to completion.
    """

    def __init__(
        self,
        n_warmup_evals: int = 3,
        n_startup_trials: int = 5,
        percentile: float = 50.0,
        direction: str = "maximize",
    ):
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")

        self.n_warmup_evals = n_warmup_evals
        self.n_startup_trials = n_startup_trials
        self.percentile = percentile
        self.direction = direction
        self.history: dict[int, list[float]] = {}
        self.num_completed_trials = 0

    def should_prune(self, eval_index: int, value: float) -> bool:
        if self.num_completed_trials < self.n_startup_trials:
            return False
        if eval_index < self.n_warmup_evals:
            return False

        past_values = self.history.get(eval_index)
        if not past_values:
            return False

        threshold = np.percentile(past_values, self.percentile)

        if self.direction == "maximize":
            return value < threshold
        return value > threshold

    def record_completed_trial(self, values_by_eval_index: list[float]) -> None:
        for eval_index, value in enumerate(values_by_eval_index):
            self.history.setdefault(eval_index, []).append(value)
        self.num_completed_trials += 1
