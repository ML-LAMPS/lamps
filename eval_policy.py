"""
Off-line evaluation of a trained policy on a given dataset.

Example usage:
    $ python eval_policy.py \
        --experiment "image-classification" \
        --eval-dataset "mtlbm/micro/set0/BCT" \
        --base-checkpoint "checkpoints/image-classification/mtrl/mtlbm/micro/set0/BCT_1/rl_model_3999968_steps.zip" \
        --objectives "model/size_billion,eval/log_loss" \
        --render "hypervolume"
"""

import argparse

from sb3_contrib.ppo_mask import MaskablePPO

from lamps.environment import EvalDatasetEnv
from lamps.settings import objectives as parse_objectives

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--eval-dataset", type=str, required=True)
parser.add_argument("--base-checkpoint", type=str, required=True)
parser.add_argument("--objectives", type=str, required=True)
parser.add_argument(
    "--render",
    type=str,
    choices=["hypervolume", "objectives"],
    default=None,
    help="Render mode (optional): hypervolume or objectives",
)


def eval_policy(
    experiment: str,
    dataset: str,
    objectives: list[str],
    policy_path: str,
    render: str | None = None,
):

    policy = MaskablePPO.load(policy_path)

    env = EvalDatasetEnv(
        experiment=experiment,
        dataset=dataset,
        objectives=parse_objectives(objectives),
    )

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, state = policy.predict(
            obs, deterministic=True, action_masks=obs["action_mask"]
        )
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

    env.render(render)

    return total_reward


def main():
    args = parser.parse_args()

    eval_policy(
        args.experiment,
        args.eval_dataset,
        args.objectives.split(","),
        args.base_checkpoint,
        render=args.render,
    )


if __name__ == "__main__":
    main()
