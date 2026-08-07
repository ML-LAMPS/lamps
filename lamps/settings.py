import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent

DEFAULT_SEED = 42

PPO_HYPERPARAMS = {
    "learning_rate": 1e-4,
    "n_steps": 4096,
    "batch_size": 256,
    "n_epochs": 15,
    "gamma": 0.99,
    "gae_lambda": 0.97,
    "clip_range": 0.20,
    "normalize_advantage": False,
    "ent_coef": 0.23837453090487948,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "stats_window_size": 100,
    "verbose": 1,
}

OBJECTIVES = {
    "model/size_billion": {"min_value": 0.0, "max_value": 10.0},
    "eval/log_loss": {"min_value": -15, "max_value": 15.0},
    "eval/neg_bleu": {"min_value": -100, "max_value": 0.0},
    "eval/neg_accuracy": {"min_value": -1.0, "max_value": 0.0},
    "eval/neg_f1_macro": {"min_value": -1.0, "max_value": 0.0},
}

OBSERVERS = (
    "lamps.observers.EpochObserver",
    "lamps.observers.RuntimeObserver",
    "lamps.observers.ActionMaskObserver",
    "lamps.observers.ModelSizeBillionObserver",
    "lamps.observers.EvalLogLossObserver",
)


def objectives(metrics: list[str]):
    return {metric: OBJECTIVES[metric] for metric in metrics}
