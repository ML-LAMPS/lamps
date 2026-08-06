from sb3_contrib.ppo_mask import MaskablePPO

from lamps.environment import EvalDatasetEnv
from lamps.settings import objectives


def eval_policy(
    experiment: str,
    dataset: str,
    objectives: dict,
    policy_path: str,
    render: str | None = None,
):

    policy = MaskablePPO.load(policy_path)

    env = EvalDatasetEnv(
        experiment=experiment,
        dataset=dataset,
        objectives=objectives,
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


if __name__ == "__main__":
    _objs = objectives(["model/size_billion", "eval/log_loss"])
    eval_policy(
        "image-classification",
        "mtlbm/micro/set0/BCT",
        _objs,
        "checkpoints/image-classification/single/mtlbm/micro/set0/BCT_1/rl_model_6000000_steps.zip",
        render="objectives",
    )
