"""
Training script for multi-task reinforcement learning (MTRL) using MaskablePPO.

Example:
    $ python train_mtrl.py --experiment "text-classification" --total-timesteps 10e6 \
        --train-datasets "all" --eval-datasets "CogComp/trec"

    $ python train_mtrl.py --experiment "machine-translation" --total-timesteps 10e6 \
        --train-datasets "all" --eval-datasets "Helsinki-NLP/opus_books[en-es]" \
        --objectives "model/size_billion,eval/log_loss"

    $ python train_mtrl.py --experiment "image-classification" --total-timesteps 10e6 \
        --train-datasets "all" --eval-datasets "mtlbm/micro/set0/BCT" \
        --objectives "model/size_billion,eval/log_loss"
"""

import os
import argparse

import numpy as np

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from lamps.environment import TrainingDatasetEnv
from lamps.repository import Repository
from lamps.utils import slugify, get_checkpoint_save_path
from lamps import settings

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--train-datasets", type=str, required=True)
parser.add_argument("--eval-datasets", type=str, required=True)
parser.add_argument("--test-datasets", type=str, default=None)
parser.add_argument("--total-timesteps", type=float, default=10e6)
parser.add_argument("--base-checkpoint", type=str, default=None)
parser.add_argument(
    "--objectives", type=str, default="model/size_billion,eval/log_loss"
)
parser.add_argument("--study-name", type=str, default="")
parser.add_argument("--tb-log-name", type=str, default=None)
parser.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
parser.add_argument("--save-path", type=str, default=None)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument(
    "--vec-env",
    type=str,
    choices=["subproc", "dummy"],
    default="subproc",
)


class MaskableTestCallback(MaskableEvalCallback):
    """MaskableEvalCallback that logs under `test/` instead of `eval/`, no checkpointing."""

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                use_masking=self.use_masking,
            )

            self.logger.record("test/mean_reward", float(np.mean(episode_rewards)))
            self.logger.record("test/mean_ep_length", float(np.mean(episode_lengths)))
            self.logger.dump(self.num_timesteps)

        return True


def build_vec_env(env_fns: list, vec_env: str):
    if vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def make_env(
    experiment: str,
    dataset: str,
    metrics: list,
):

    objectives = {metric: settings.OBJECTIVES[metric] for metric in metrics}

    def _init():
        env_kwargs = {
            "experiment": experiment,
            "dataset": dataset,
            "objectives": objectives,
        }
        env = TrainingDatasetEnv(**env_kwargs)
        env = Monitor(env)

        return env

    return _init


def parse_train_eval_datasets(args):
    train_datasets = []
    eval_datasets = []
    test_datasets = []

    all_datasets = Repository.list_datasets(experiment=args.experiment)

    if args.train_datasets == "all":
        train_datasets = all_datasets.copy()
    else:
        train_datasets = []

        for ds_entry in args.train_datasets.split(","):
            num_envs = 1

            if ":" in ds_entry:
                ds_entry, num_envs = ds_entry.split(":")
                num_envs = int(num_envs)

            train_datasets.extend([ds_entry] * num_envs)

    if args.eval_datasets == "all":
        eval_datasets = all_datasets.copy()
    else:
        eval_datasets = [ds.strip() for ds in args.eval_datasets.split(",")]

    if args.test_datasets:
        if args.test_datasets == "all":
            test_datasets = all_datasets.copy()
        else:
            test_datasets = [ds.strip() for ds in args.test_datasets.split(",")]

    # Ensure no overlap between train and eval/test datasets
    if args.train_datasets == "all":
        train_datasets = [
            ds
            for ds in train_datasets
            if ds not in eval_datasets and ds not in test_datasets
        ]

    if args.eval_datasets == "all":
        eval_datasets = [ds for ds in eval_datasets if ds not in train_datasets]

    return train_datasets, eval_datasets, test_datasets


def parse_tb_log_name(args, train_datasets, eval_datasets):
    log_name = "train="

    if args.train_datasets == "all":
        log_name += "all"
    else:
        _train_datasets = [slugify(ds) for ds in set(train_datasets)]
        log_name += ",".join(_train_datasets)

    log_name += " | eval="

    if args.eval_datasets == "all":
        log_name += "all"
    else:
        _eval_datasets = [slugify(ds) for ds in eval_datasets]
        log_name += ",".join(_eval_datasets)

    return log_name


def main():
    args = parser.parse_args()

    metrics = args.objectives.split(",")

    train_datasets, eval_datasets, test_datasets = parse_train_eval_datasets(args)

    if args.save_path is not None:
        save_path = args.save_path
        _tb_logs_sufix = os.path.relpath(
            save_path, os.path.join("checkpoints", args.experiment)
        )
    else:
        _tb_logs_sufix = os.path.join("mtrl", args.study_name)
        save_path = get_checkpoint_save_path(
            args.experiment, eval_datasets[0], _tb_logs_sufix
        )

    tensorboard_log = f"./tb_logs/{args.experiment}/{_tb_logs_sufix}"

    print(f"Training datasets: {train_datasets}")
    print(f"Evaluation datasets: {eval_datasets}")
    if test_datasets:
        print(f"Test datasets: {test_datasets}")

    if args.tb_log_name:
        tb_log_name = args.tb_log_name
    else:
        tb_log_name = parse_tb_log_name(args, train_datasets, eval_datasets)

    train_envs = build_vec_env(
        [
            make_env(
                args.experiment,
                dataset,
                metrics,
            )
            for dataset in train_datasets
        ],
        args.vec_env,
    )
    eval_env = build_vec_env(
        [
            make_env(
                args.experiment,
                dataset,
                metrics,
            )
            for dataset in eval_datasets
        ],
        args.vec_env,
    )

    callback_list = [
        CheckpointCallback(
            save_freq=int(args.total_timesteps / len(train_datasets)) // 10,
            save_path=save_path,
            verbose=1,
        ),
        MaskableEvalCallback(eval_env=eval_env),
    ]

    if test_datasets:
        test_env = build_vec_env(
            [
                make_env(
                    args.experiment,
                    dataset,
                    metrics,
                )
                for dataset in test_datasets
            ],
            args.vec_env,
        )
        callback_list.append(MaskableTestCallback(eval_env=test_env))

    callbacks = CallbackList(callback_list)

    if args.base_checkpoint is not None:
        print(f"Loading base checkpoint from {args.base_checkpoint}")
        model = MaskablePPO.load(
            args.base_checkpoint,
            env=train_envs,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )
    else:
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=train_envs,
            device=args.device,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            **settings.PPO_HYPERPARAMS,
        )

    model.learn(
        total_timesteps=int(args.total_timesteps),
        progress_bar=True,
        callback=callbacks,
        tb_log_name=tb_log_name,
    )


if __name__ == "__main__":
    main()
