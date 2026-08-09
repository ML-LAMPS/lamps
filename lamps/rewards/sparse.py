import numpy as np

from .base import BaseReward

OPTIMAL_REWARD = 3000.0


class SparseReward(BaseReward):
    """
    Terminal-only reward: zero everywhere except at episode termination, where
    it rewards faster discovery of the Pareto-optimal set (Eq. 6 in the LAMPS
    paper). Policy-invariant by construction (Lemma C.1).
    """

    NAME = "sparse"

    def compute(self, terminated: bool, valid_mask: np.ndarray | None = None) -> float:
        if not terminated:
            return 0.0

        env = self.env
        num_pareto_models = len(env.pareto_models)

        if valid_mask is None:
            valid_mask = env.valid_mask

        num_completed_models = env.num_models - int(np.count_nonzero(valid_mask))

        _reward = (env.num_models - num_completed_models) / env.search_time()
        _optimal = (env.num_models - num_pareto_models) / env.optimal_runtime

        return OPTIMAL_REWARD * (_reward / _optimal)
