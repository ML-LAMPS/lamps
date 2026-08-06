from typing import Any

import warnings

import matplotlib.pyplot as plt

from lamps.environment.base import DatasetEnv


class EvalDatasetEnv(DatasetEnv):
    """
    Used to evaluate a trained policy against repository data. Tracks the
    hypervolume/search-time trajectory of the episode so it can be rendered.
    """

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = super().reset(seed=seed, options=options)

        self.history = {"time": [0.0], "hypervolume": [0.0]}

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        self.history["time"].append(self.search_time())
        self.history["hypervolume"].append(self.hypervolume(only_finished=True))

        return observation, reward, terminated, truncated, info

    def render(self, mode=None):
        if mode is None:
            return

        if mode == "hypervolume":
            self._render_hypervolume()
        elif mode == "objectives":
            self._render_objectives()
        else:
            raise ValueError(
                f"Unknown render mode: {mode!r}. Expected 'hypervolume' or 'objectives'."
            )

    def _render_hypervolume(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        ax.plot(self.history["time"], self.history["hypervolume"], marker="o")

        if self.optimal_hv is not None:
            ax.axhline(
                self.optimal_hv, color="gray", linestyle="--", label="Optimal HV"
            )

        ax.set_xlabel("Search time (s)")
        ax.set_ylabel("Hypervolume")
        ax.set_title(f"{self.dataset} — policy evaluation")
        ax.legend()

        plt.show()

    def _current_objective_values(self):
        return {
            objective: [
                self.repository.get_metric(model, objective, self.epoch_counts[i])
                for i, model in enumerate(self.models)
            ]
            for objective in self.objectives
        }

    def _render_objectives(self):

        num_objectives = len(self.objectives)
        values = self._current_objective_values()

        if num_objectives == 1:
            objective = self.objectives[0]

            _, ax = plt.subplots()
            ax.bar(self.models, values[objective])
            ax.set_xlabel("Model")
            ax.set_ylabel(objective)
            ax.set_title(f"{self.dataset} — {objective}")
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
            plt.tight_layout()

        elif num_objectives in (2, 3):
            obj_x, obj_y, *obj_z = self.objectives
            x, y = values[obj_x], values[obj_y]

            fig = plt.figure()
            if obj_z:
                ax = fig.add_subplot(projection="3d")
                z = values[obj_z[0]]
                ax.scatter(x, y, z)  # type: ignore[arg-type]  # Axes3D overload not resolved from projection="3d"
                for model, xi, yi, zi in zip(self.models, x, y, z):
                    ax.text(xi, yi, zi, model, fontsize=8)
                ax.set_zlabel(obj_z[0])
            else:
                ax = fig.add_subplot()
                ax.scatter(x, y)
                for model, xi, yi in zip(self.models, x, y):
                    ax.annotate(model, (xi, yi), fontsize=8)

            ax.set_xlabel(obj_x)
            ax.set_ylabel(obj_y)
            ax.set_title(f"{self.dataset} — objectives")

        else:
            warnings.warn(
                f"Cannot render {num_objectives} objectives; only 1-3 are supported."
            )
            return

        plt.show()
