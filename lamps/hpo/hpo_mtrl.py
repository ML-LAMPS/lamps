"""
Multi-objective hyperparameter optimization for train_mtrl.py using Optuna.

The base observers in settings.OBSERVERS (epochs, runtime, action mask, and
the two objective-value observers) are held fixed - removing any of those
would just blind the policy to something it needs. On top of that, four
independently-optional feature observers (ParetoDominanceObserver,
LogLossDynamicsObserver, PerModelLogLossGPObserver, CrowdingDistanceObserver)
are searched as independent booleans, alongside PPO optimization
hyperparameters, the reward function (and its own hyperparameters), and
policy network architecture. The study optimizes two objectives jointly
rather than a single scalarized score:

    1. maximize: mean validation reward over the trailing evaluations
    2. minimize: instability = std(trailing validation reward)
                              + max(0, peak_val_reward - trailing_mean)

The second term is the study's "generalization stability" objective: std
alone doesn't distinguish healthy noise from a policy that overfits the
training distribution and degrades on held-out data, so an explicit
peak-vs-tail gap is added on top of it. Test datasets are never touched by
the study - only used, once, to report the final chosen configuration.

Note: Optuna's Trial.report()/should_prune() raise NotImplementedError for
multi-objective studies (verified against optuna==4.9.0), so there is no
built-in mid-training pruning here. lamps.hpo.ManualMedianPruner reimplements
the same idea by hand, using only optuna.TrialPruned (which multi-objective
studies do support), to keep the search from spending a full trial's budget
on hyperparameters that are already trailing.

Note: PerModelLogLossGPObserver requires a precomputed GP-posterior cache on
disk per dataset (see
lamps/observers/per_model_log_loss_gp/precompute_independent_loss.py); a
trial that samples it in for a dataset without that cache will raise
FileNotFoundError. study.optimize() below is run with catch=(Exception,) so
that - or any other single-trial failure - fails just that trial rather than
ending the whole sweep.

Example (run from the repo root, so train_mtrl.py stays importable):
    $ python -m lamps.hpo.hpo_mtrl --experiment "image-classification" \
        --train-datasets "all" --val-datasets "mtlbm/micro/set0/BCT" \
        --n-trials 50 --timesteps-per-trial 2e6
"""

import argparse

import numpy as np
import optuna

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from lamps import settings

from train_mtrl import build_vec_env, make_env, parse_datasets

from .pruning import ManualMedianPruner

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--train-datasets", type=str, required=True)
parser.add_argument("--val-datasets", type=str, required=True)
parser.add_argument("--test-datasets", type=str, default=None)
parser.add_argument(
    "--objectives", type=str, default="model/size_billion,eval/log_loss"
)
parser.add_argument("--n-trials", type=int, default=50)
parser.add_argument("--timesteps-per-trial", type=float, default=2e6)
parser.add_argument("--eval-freq", type=int, default=10000)
parser.add_argument("--n-eval-episodes", type=int, default=20)
parser.add_argument("--tail-evals", type=int, default=10)
parser.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
parser.add_argument(
    "--sampler-seed",
    type=int,
    default=None,
    help=(
        "Seed for the Optuna sampler only. Leave unset (default) when running "
        "multiple independent workers against the same --storage, so each "
        "worker's sampler RNG diverges instead of proposing identical trials."
    ),
)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument(
    "--vec-env", type=str, choices=["subproc", "dummy"], default="dummy"
)
parser.add_argument("--study-name", type=str, default="lamps-hpo")
parser.add_argument("--storage", type=str, default=None)
parser.add_argument("--tb-logs-dir", type=str, default="tb_logs_hpo")


class TrialEvalCallback(MaskableEvalCallback):
    """Records per-eval validation reward for the HPO objective and logs it to TB."""

    def __init__(self, *args, pruner: ManualMedianPruner, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruner = pruner
        self.history: list[float] = []
        self.diverged = False
        self.pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                use_masking=self.use_masking,
            )
            mean_reward = float(np.mean(episode_rewards))

            if not np.isfinite(mean_reward):
                self.diverged = True
                return False

            self.logger.record("val/mean_reward", mean_reward)
            self.logger.record("val/mean_ep_length", float(np.mean(episode_lengths)))
            self.logger.dump(self.num_timesteps)

            eval_index = len(self.history)

            if self.pruner.should_prune(eval_index, mean_reward):
                self.pruned = True
                return False

            self.history.append(mean_reward)

        return True


OPTIONAL_OBSERVERS = (
    "lamps.observers.ParetoDominanceObserver",
    "lamps.observers.LogLossDynamicsObserver",
    "lamps.observers.PerModelLogLossGPObserver",
    "lamps.observers.CrowdingDistanceObserver",
)


def sample_observers(trial: optuna.Trial) -> list[str]:
    observers: list[str] = list(settings.OBSERVERS)

    for observer in OPTIONAL_OBSERVERS:
        name = observer.rsplit(".", 1)[-1]
        if trial.suggest_categorical(f"use_{name}", [True, False]):
            observers.append(observer)

    return observers


def sample_reward(trial: optuna.Trial) -> tuple[str, dict]:
    reward_name = trial.suggest_categorical("reward", ["sparse", "potential_shaped"])

    if reward_name == "sparse":
        return "lamps.rewards.SparseReward", {}

    shaping_scale = trial.suggest_float("shaping_scale", 0.005, 0.1, log=True)
    return "lamps.rewards.PotentialShapedReward", {"shaping_scale": shaping_scale}


def sample_ppo_kwargs(trial: optuna.Trial) -> dict:
    return {
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "normalize_advantage": trial.suggest_categorical(
            "normalize_advantage", [True, False]
        ),
    }


def sample_policy_kwargs(trial: optuna.Trial) -> dict:
    net_arch_sizes = {
        "small": [64, 64],
        "medium": [128, 128],
        "large": [256, 256],
    }[trial.suggest_categorical("net_arch", ["small", "medium", "large"])]

    separate_networks = trial.suggest_categorical("separate_networks", [True, False])
    net_arch = (
        dict(pi=net_arch_sizes, vf=net_arch_sizes)
        if separate_networks
        else net_arch_sizes
    )

    optimizer_kwargs = {}
    if trial.suggest_categorical("use_weight_decay", [True, False]):
        optimizer_kwargs["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        )

    return {
        "net_arch": net_arch,
        "share_features_extractor": trial.suggest_categorical(
            "share_features_extractor", [True, False]
        ),
        "optimizer_kwargs": optimizer_kwargs,
    }


def compute_objectives(history: list[float], tail_evals: int) -> tuple[float, float]:
    if not history:
        return float("-1e6"), float("1e6")

    tail = history[-tail_evals:] if len(history) >= tail_evals else history
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))
    overfitting_gap = max(0.0, float(np.max(history)) - tail_mean)

    return tail_mean, tail_std + overfitting_gap


class Objective:
    def __init__(self, args: argparse.Namespace, pruner: ManualMedianPruner):
        self.args = args
        self.pruner = pruner
        self.metrics = args.objectives.split(",")
        self.train_datasets, self.val_datasets, _ = parse_datasets(args)

    def __call__(self, trial: optuna.Trial) -> tuple[float, float]:
        args = self.args
        reward, reward_kwargs = sample_reward(trial)
        ppo_kwargs = sample_ppo_kwargs(trial)
        policy_kwargs = sample_policy_kwargs(trial)
        observers = sample_observers(trial)

        train_envs = None
        val_env = None

        try:
            train_envs = build_vec_env(
                [
                    make_env(
                        args.experiment, dataset, self.metrics, reward, reward_kwargs, observers
                    )
                    for dataset in self.train_datasets
                ],
                args.vec_env,
            )
            val_env = build_vec_env(
                [
                    make_env(
                        args.experiment, dataset, self.metrics, reward, reward_kwargs, observers
                    )
                    for dataset in self.val_datasets
                ],
                args.vec_env,
            )

            model = MaskablePPO(
                policy="MultiInputPolicy",
                env=train_envs,
                device=args.device,
                seed=args.seed,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"{args.tb_logs_dir}/{args.experiment}/{args.study_name}",
                **{**settings.PPO_HYPERPARAMS, **ppo_kwargs, "verbose": 0},
            )

            callback = TrialEvalCallback(
                eval_env=val_env,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                pruner=self.pruner,
            )

            model.learn(
                total_timesteps=int(args.timesteps_per_trial),
                tb_log_name=f"trial_{trial.number}",
                callback=callback,
                progress_bar=False,
            )
        finally:
            if train_envs is not None:
                train_envs.close()
            if val_env is not None:
                val_env.close()

        if callback.diverged:
            raise optuna.TrialPruned("Validation reward diverged (non-finite).")

        if callback.pruned:
            raise optuna.TrialPruned("Below the historical median at an early checkpoint.")

        self.pruner.record_completed_trial(callback.history)

        return compute_objectives(callback.history, args.tail_evals)


def main():
    args = parser.parse_args()

    pruner = ManualMedianPruner()
    objective = Objective(args, pruner)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=args.sampler_seed),
    )
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    print(f"\nPareto front ({len(study.best_trials)} trials):")
    for trial in study.best_trials:
        print(
            f"  trial={trial.number} "
            f"tail_mean_reward={trial.values[0]:.2f} "
            f"instability={trial.values[1]:.2f} "
            f"params={trial.params}"
        )


if __name__ == "__main__":
    main()
