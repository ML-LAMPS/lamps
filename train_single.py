"""
Train a single-task PPO model on a specified dataset.

Example:
    $ python train_single.py --experiment "text-classification" \
        --dataset "CogComp/trec" \
        --objectives "model/size_billion,eval/neg_bleu" \
        --total-timesteps 10e6

    $ python train_single.py --experiment "machine-translation" \
        --dataset "Helsinki-NLP/opus_books[en-es]" \
        --objectives "model/size_billion,eval/neg_bleu" \
        --total-timesteps 10e6

    $ python train_single.py --experiment "image-classification" \
        --dataset "mtlbm/micro/set0/BRD" \
        --objectives "model/size_billion,eval/log_loss" \
        --total-timesteps 10e6

To follow the training progress, use TensorBoard:
    $ tensorboard --logdir ./tb_logs
"""

import argparse

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from lamps.environment import TrainingDatasetEnv
from lamps.utils import slugify, get_checkpoint_save_path
from lamps import settings

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--total-timesteps", type=float, default=10e6)
parser.add_argument(
    "--objectives", type=str, default="model/size_billion,eval/log_loss"
)
parser.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
parser.add_argument("--base-checkpoint", type=str, default=None)
parser.add_argument("--device", type=str, default="auto")


def make_env(experiment: str, dataset: str, metrics: list):
    objectives = {
        metric: settings.OBJECTIVES[metric]
        for metric in settings.OBJECTIVES
        if metric in metrics
    }

    def _init():
        env = TrainingDatasetEnv(
            experiment=experiment,
            dataset=dataset,
            objectives=objectives,
        )
        env = Monitor(env)

        return env

    return _init


def main():

    args = parser.parse_args()

    tensorboard_log = f"./tb_logs/{args.experiment}/single"
    save_path = get_checkpoint_save_path(args.experiment, args.dataset, "single")

    metrics = args.objectives.split(",")

    callbacks = [
        CheckpointCallback(
            save_freq=int(args.total_timesteps / args.num_envs) // 10,
            save_path=save_path,
            verbose=1,
        ),
    ]

    if args.num_envs == 1:
        env = make_env(args.experiment, args.dataset, metrics)()
    else:
        env = SubprocVecEnv(
            [make_env(args.experiment, args.dataset, metrics)] * args.num_envs,
            start_method="spawn",
        )
        eval_env = SubprocVecEnv(
            [make_env(args.experiment, args.dataset, metrics)],
            start_method="spawn",
        )
        callbacks.append(MaskableEvalCallback(eval_env=eval_env))

    if args.base_checkpoint is not None:
        print(f"Loading base checkpoint from {args.base_checkpoint}")
        model = MaskablePPO.load(
            args.base_checkpoint,
            env=env,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )
    else:
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=env,
            device=args.device,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            **settings.PPO_HYPERPARAMS,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=CallbackList(callbacks),
        tb_log_name=slugify(args.dataset),
    )


if __name__ == "__main__":
    main()
