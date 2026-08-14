# Hyperparameter Optimization (HPO)

This package holds `hpo_mtrl.py`, the Optuna study that searches for
PPO/reward/network/observer hyperparameters that are both **stable** and
**high-performing** on held-out validation data.

## What's in `lamps/hpo/`

- `hpo_mtrl.py` → the runnable study. It imports `make_env` / `build_vec_env`
  / `parse_datasets` from `train_mtrl.py` at the repo root, so it must be run
  as a module (`python -m lamps.hpo.hpo_mtrl`, see below) — running it via a
  bare file path breaks that import, since the script's own directory (not
  the repo root) would end up on `sys.path` instead.
- `pruning.py` → `ManualMedianPruner`. Optuna's `Trial.report()` /
  `Trial.should_prune()` raise `NotImplementedError` for multi-objective
  studies (verified against `optuna==4.9.0`), so there's no built-in way to
  stop a clearly-bad trial partway through training once you ask for a
  Pareto front. This reimplements the same idea by hand: a trial is pruned
  if, at a given validation checkpoint, its value is worse than the median
  of same-checkpoint values from trials that already ran to completion
  ("worse" respects a `direction="maximize"|"minimize"` setting — `hpo_mtrl.py`
  uses `"minimize"`, since it prunes on episode length, see below). Pruned
  trials never pollute that historical baseline for later trials.

## What's searched

| Group | Params |
|---|---|
| PPO stability | `ent_coef`, `learning_rate`, `clip_range`, `gae_lambda`, `normalize_advantage` |
| Reward | `SparseReward` vs. `PotentialShapedReward` (+ `shaping_scale` if the latter) |
| Network | `net_arch` size (small/medium/large), shared vs. separate pi/vf networks, `share_features_extractor`, optional `weight_decay` |
| Observers | Four independent booleans: `ParetoDominanceObserver`, `LogLossDynamicsObserver`, `PerModelLogLossGPObserver`, `CrowdingDistanceObserver` — see below before including the third one |

`settings.OBSERVERS` (epochs, runtime, action mask, and the two
objective-value observers) stays fixed underneath these four — those aren't
optional, `DatasetEnv` depends on some of them structurally and the others
are literally what the reward is scored on, so there's no real hypothesis to
test by removing them. The four above are independently-developed features
with a genuine "does this help or hurt" question behind each, so they're
searched as independent booleans rather than combinatorially — same
reasoning as the hyperparameters above: trust the sampler over enumerating
all 16 combinations.

`gamma` is deliberately **not** searched: `PotentialShapedReward`'s shaping
term assumes the same discount as PPO's own `gamma`, so tuning one without
the other would silently break the policy-invariance guarantee the reward
relies on.

### Prerequisite: precompute the GP-posterior cache

`PerModelLogLossGPObserver` needs a precomputed cache file per dataset (see
`lamps/observers/per_model_log_loss_gp/`). A trial that samples it in for a
dataset without that cache raises `FileNotFoundError` during env
construction — `study.optimize()` is run with `catch=(Exception,)`, so that
just fails the one trial rather than the whole sweep, but you'd still rather
not waste trials that way. Build the cache once, for every dataset in the
experiment, before running the sweep:

```bash
python -m lamps.observers.per_model_log_loss_gp.precompute_independent_loss \
  --experiment image-classification \
  --datasets "all"
```

Run this from the repo root too, so the cache lands at the default
`.cache/per_model_log_loss_gp/` path both this script and the observer agree
on without needing to pass `cache_dir` explicitly anywhere.

## Requirements

Already pinned in `requirements.txt`: `optuna==4.9.0`,
`optuna-dashboard==0.20.0`. Install with `pip install -r requirements.txt` if
you haven't already.

## Running a sweep

First, build the GP-posterior cache once for the experiment (see above —
this is what lets `use_PerModelLogLossGPObserver: True` trials succeed
instead of failing):

```bash
python -m lamps.observers.per_model_log_loss_gp.precompute_independent_loss \
  --experiment image-classification \
  --datasets "all"
```

Then use **persistent storage** (SQLite is enough for a single-machine run)
so you can watch progress live and resume later — the default in-memory
storage disappears when the process exits and can't be attached to from
another process:

```bash
python -m lamps.hpo.hpo_mtrl \
  --experiment image-classification \
  --train-datasets "all" \
  --val-datasets "mtlbm/micro/set0/BCT" \
  --n-trials 50 \
  --timesteps-per-trial 2e6 \
  --study-name lamps-hpo \
  --storage "sqlite:///hpo_mtrl.db"
```

```bash
PARTITION="haswell-64"
# PARTITION="haswell-256"
# PARTITION="epyc-256"
# PARTITION="epyc-128"
# PARTITION="epyc-384"
# PARTITION="epyc-768"
# PARTITION="skylake-96"
# PARTITION="skylake-384"

sbatch -J RUN1 --partition=$PARTITION --cpus-per-task=16 --mem=24G scripts/run.sbatch \
python -m lamps.hpo.hpo_mtrl \
  --experiment image-classification \
  --train-datasets "all" \
  --val-datasets "mtlbm/micro/set0/BCT" \
  --n-trials 25 \
  --timesteps-per-trial 10e6 \
  --study-name lamps-hpo \
  --storage "sqlite:///hpo_mtrl.db"
```

Run this **from the repo root** (same directory as `train_mtrl.py`) using
`-m`, not a file path — `--storage`/`--tb-logs-dir` paths and `make_env`'s
repository lookups are relative to the repo root, and `-m` is what keeps
`train_mtrl.py` importable from a script that now lives two directories
deeper (`python lamps/hpo/hpo_mtrl.py ...` will fail with
`ModuleNotFoundError: No module named 'lamps'`).

### Running independent parallel workers

To speed the search up, launch several of the `sbatch` jobs above at once —
all pointed at the **same `--storage` and `--study-name`** — and Optuna
coordinates them through that shared file (`load_if_exists=True` means every
worker joins the same study instead of starting a fresh one).

The one thing to get right: **don't pass `--sampler-seed`.** It defaults to
unset (OS-random per process) specifically so each independent worker's
sampler diverges. If you fix it to the same number across workers (or reuse
`--seed`, which the script used to do before this was split out), every
worker's sampler starts from an identical RNG state and — since that state
isn't synchronized through storage, only completed trials are — they'll
each propose the *same* early hyperparameters instead of exploring
different ones, defeating the point of running them in parallel. `--seed`
(the PPO training seed) is fine to leave fixed across workers; that's a
different concern and keeping it constant makes trials comparable on
hyperparameters alone rather than on training-run luck.

SQLite handles a modest number of concurrent workers fine for this use case
(each trial only writes once, at completion or pruning). If you scale up to
dozens of simultaneous workers and see write-lock contention, that's the
point to move `--storage` to a real RDB (Postgres/MySQL) instead.

### Key arguments

| Flag | Default | Notes |
|---|---|---|
| `--train-datasets` / `--val-datasets` | required | Same syntax as `train_mtrl.py` (comma-separated, `"all"`, or `name:num_envs`). `--test-datasets` is accepted but only for symmetry with `parse_datasets`'s exclusion logic — the study never evaluates on it. |
| `--n-trials` | `50` | Number of Optuna trials in this run. |
| `--timesteps-per-trial` | `2e6` | Reduced training budget per trial — this is the main compute lever, not the (unavailable) mid-training pruning. Pick something short enough to sweep many trials, but long enough for the validation curve to say something meaningful. |
| `--eval-freq` | `10000` | Steps between validation evaluations (used for the objective *and* the manual pruner's checkpoints). |
| `--disable-pruning` | off | Skip the manual median pruner entirely - every trial runs to `--timesteps-per-trial` regardless of how it compares to the historical median. |
| `--n-eval-episodes` | `20` | Episodes per validation eval. Kept well above SB3's own default of 5 — with only a handful of episodes, the objective is mostly measuring sampling noise, not the policy. |
| `--tail-evals` | `10` | How many trailing eval points define "converged" performance for the objectives (see below). |
| `--vec-env` | `dummy` | Deliberately different from `train_mtrl.py`'s `subproc` default — spawning a fresh process pool per trial across dozens of trials is mostly overhead. Switch to `subproc` only if your envs are slow enough to need it. |
| `--study-name` / `--storage` | `lamps-hpo` / `None` | Set `--storage` to something like `sqlite:///hpo_mtrl.db` to persist and enable the dashboard. |
| `--sampler-seed` | `None` | Seed for the Optuna sampler *only*. Leave unset when running multiple parallel workers (see above) — that's what makes their sampling diverge instead of colliding. Set it explicitly only for a single-worker run you want to be reproducible. |
| `--seed` | `settings.DEFAULT_SEED` | PPO/env training seed, independent of `--sampler-seed`. Fine to keep fixed across workers. |
| `--tb-logs-dir` | `tb_logs_hpo` | Base folder for per-trial TensorBoard logs (kept separate from `train_mtrl.py`'s `tb_logs`). Each trial writes to `{tb_logs_dir}/{experiment}/{study_name}/trial_{N}_1/`. |

## Watching progress

**TensorBoard** (per-trial training curves + validation reward/episode length):

```bash
tensorboard --logdir tb_logs_hpo
```

**Optuna dashboard** (cross-trial comparison and the Pareto front — the more
useful view for this study):

```bash
optuna-dashboard sqlite:///hpo_mtrl.db
```

Then open http://127.0.0.1:8080. You can launch this *while the sweep is
still running* — it reads from the same SQLite file the study is writing to,
so trials appear as they finish. Since this is a multi-objective study, the
dashboard's Pareto-front / scatter plot across the two objectives is the
view to use, not a single optimization-history curve.

## Understanding the results

The study optimizes two objectives jointly rather than one scalarized score,
so there's no single "best" trial — `study.best_trials` (printed at the end
of a run, and shown in the dashboard) is the **Pareto front**: trials where
you can't improve one objective without making the other worse.

Both objectives are computed from **validation episode length**, not
validation reward. This matters because the reward function itself
(`SparseReward` vs. `PotentialShapedReward`, with its own `shaping_scale`)
is one of the things being searched — its scale isn't consistent across
trials, so comparing raw reward across trials that used different reward
functions is comparing different units. Episode length has no such problem:
shorter always means "found the Pareto-optimal set faster," regardless of
which reward trained the policy to get there.

1. **minimize** — mean validation episode length over the trailing
   `--tail-evals` evaluations (not a single best sample, which would just
   reward a lucky episode).
2. **minimize** — instability, defined as
   `std(trailing episode length) + max(0, trailing_mean - best_ep_length)`.
   The `std` term catches raw oscillation; the second term specifically
   penalizes a policy that trains down to a short episode length and then
   degrades (gets slower) on held-out data — the overfitting pattern this
   whole study exists to avoid, which `std` alone wouldn't distinguish from
   healthy noise.

Pick a point off the Pareto front based on how much peak performance you're
willing to trade for stability — that judgment call is exactly why this
wasn't collapsed into one scalarized number.

## Resuming or extending a study

Rerun the same command with the same `--study-name` and `--storage` (and a
larger `--n-trials` if you want more) — `load_if_exists=True` means Optuna
picks up the existing trials rather than starting over, and the sampler and
manual pruner both benefit from the extra history.

## Applying a winning configuration

`train_mtrl.py` doesn't yet read hyperparameters from a file — a winning
trial's `params` need to be applied by hand before a full verification run:

- `ent_coef`, `learning_rate`, `clip_range`, `gae_lambda`,
  `normalize_advantage` → update `lamps/settings.py`'s `PPO_HYPERPARAMS`.
- `reward` → update `lamps/settings.py`'s `REWARD` to
  `"lamps.rewards.SparseReward"` or `"lamps.rewards.PotentialShapedReward"`.
- `use_ParetoDominanceObserver`, `use_LogLossDynamicsObserver`,
  `use_PerModelLogLossGPObserver`, `use_CrowdingDistanceObserver` →
  uncomment (or leave commented) the matching line in `lamps/settings.py`'s
  `OBSERVERS` tuple to match each boolean.
- `shaping_scale`, and the network architecture params (`net_arch`,
  `separate_networks`, `share_features_extractor`, `weight_decay`) aren't
  wired into `train_mtrl.py`'s CLI or `settings.py` at all yet — applying
  these currently means passing them manually where `train_mtrl.py`
  constructs the env/model (`reward_kwargs` on `make_env`, `policy_kwargs`
  on `MaskablePPO`). If you end up doing this often, that's worth wiring up
  properly rather than hand-editing the script each time.

As discussed when this study was designed: verify the 2-3 most promising
Pareto-front trials at full training budget (and ideally another held-out
fold) before picking a final configuration — a config that looks good on one
validation split's reduced-budget run isn't guaranteed to hold up further out.
