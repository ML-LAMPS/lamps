import numpy as np

from lamps import settings

from .sparse import SparseReward, OPTIMAL_REWARD


def potential_shaping(
    potential_before: float, potential_after: float, terminated: bool, gamma: float
) -> float:
    """Policy-invariant shaping term F_t = gamma*Phi(s_{t+1}) - Phi(s_t); Phi(terminal)=0."""
    next_potential = 0.0 if terminated else potential_after
    return gamma * next_potential - potential_before


class PotentialShapedReward(SparseReward):
    """
    SparseReward plus potential-based shaping (Ng, Harada & Russell, 1999) on
    epochs invested in Pareto-optimal models. Gives denser per-step credit
    without changing the optimal policy under the discounted return, since
    the reward is zero everywhere except at termination (see SparseReward).
    """

    NAME = "potential_shaped"

    def __init__(
        self,
        env,
        gamma: float | None = None,
        shaping_scale: float = 0.02,
    ):
        super().__init__(env)
        self.gamma = gamma if gamma is not None else settings.PPO_HYPERPARAMS["gamma"]
        self.shaping_scale = shaping_scale
        self.pareto_epoch_budget = int(
            env.available_epochs[env.pareto_models_idx].sum()
        )
        self._potential_before = 0.0
        self._last_unshaped = 0.0
        self._last_shaping = 0.0

    def reset(self):
        self._potential_before = 0.0
        self._last_unshaped = 0.0
        self._last_shaping = 0.0

    def before_step(self, action):
        self._potential_before = self.potential()

    def compute(self, terminated: bool, valid_mask: np.ndarray | None = None) -> float:
        unshaped = super().compute(terminated, valid_mask)
        potential_after = 0.0 if terminated else self.potential()
        shaping = potential_shaping(
            self._potential_before, potential_after, terminated, self.gamma
        )

        self._last_unshaped = unshaped
        self._last_shaping = shaping

        return unshaped + shaping

    def info(self) -> dict:
        return {
            "reward/unshaped": self._last_unshaped,
            "reward/shaping": self._last_shaping,
        }

    def potential(self) -> float:
        env = self.env
        pareto_progress = int(env.epoch_counts[env.pareto_models_idx].sum())
        return (
            self.shaping_scale
            * OPTIMAL_REWARD
            * pareto_progress
            / self.pareto_epoch_budget
        )
