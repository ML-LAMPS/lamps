# LAMPS: Landmark-Guided Policy Optimization for Multi-Objective Language Model Selection

Official implementation of **LAMPS** (LAnguage Model Pareto Selection), a multi-objective AutoML
framework that meta-learns a resource-allocation policy to efficiently identify the Pareto front of
candidate pretrained LLMs for a task-specific dataset. LAMPS combines:

- **Landmark fine-tuning** — early performance indicators obtained by fine-tuning models on
  exponentially larger subsets of the training data.
- **Meta-learned resource allocation** — a PPO policy, trained with invalid action masking over a
  meta-dataset of historical fine-tuning trajectories, that decides which model to train next.

> Monteiro, M., Li, W., Wang, P., Kloft, M., and Fellenz, S. *Landmark-Guided Policy Optimization
> for Multi-Objective Language Model Selection.* ICML 2026.

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## The meta-dataset

LAMPS trains and evaluates by **replaying** pre-recorded landmark fine-tuning curves rather than
fine-tuning models on the fly. `Repository` (`lamps/repository.py`) expects one directory per
(model, dataset) pair under `metadataset/<experiment>/`, containing TensorBoard event files:

```
metadataset/<experiment>/m=<model>__d=<dataset>__s=<split>__lr=<lr>__bs=<bs>/
    events.out.tfevents...
```

where `/` in model and dataset names is escaped as `_SL_`, and `eval/*` scalars (loss, BLEU,
epoch runtime, etc.) are logged per epoch. Every model referenced must also have an entry in
`model_sizes.json` (parameter count).

`metadataset/` is not committed to this repository (see `.gitignore`) because of its size. It is
distributed separately as a zip archive on Google Drive:

**https://drive.google.com/file/d/1V33DuwFnoZtl0cf3imQ8SvNQHqgUy90M/view?usp=sharing**

The archive already contains the `metadataset/` folder at its root (with `text-classification/`,
`machine-translation/`, etc. inside), so extract it directly into the repository root.

1. Open the link above in a browser and click **Download**.
2. Move the downloaded `metadataset.zip` into the repository root.
3. Extract it and remove the archive (identical on macOS and Linux):

   ```bash
   unzip metadataset.zip -d .
   rm metadataset.zip
   ```

After extraction, use `Repository.list_datasets(experiment)` / `Repository.list_models(experiment)`
to check what a given `metadataset/<experiment>/` directory contains.

## Reproducing the main results

### Multi-task meta-training (leave-one-out, Section 6)

Trains a single policy across all-but-one dataset and evaluates zero-shot transfer to the held-out
dataset — this is the protocol used to produce Figures 3–4 and Tables 1–2:

```bash
python train_mtrl.py \
    --experiment "text-classification" \
    --train-datasets "all" \
    --eval-datasets "CogComp/trec" \
    --objectives "model/size_billion,eval/log_loss" \
    --total-timesteps 20e6
```

> `--train-datasets "all"` trains on every dataset in the experiment, but `train_mtrl.py`
> automatically removes anything listed in `--eval-datasets` from that set before training starts.
> The held-out dataset is therefore never seen during meta-training, so there is no leakage
> between the reported held-out result and the training set.

Repeat with each dataset in turn as `--eval-datasets` to reproduce the full leave-one-out sweep
(Table 1: two objectives; Table 2: add `eval/neg_bleu` to `--objectives` for the three-objective
machine translation setting).

To reproduce the single-objective ablation (Table 3, Appendix D.2), pass a single metric, e.g.
`--objectives "eval/log_loss"`.

### Monitoring training

```bash
tensorboard --logdir ./tb_logs
```

Checkpoints are written to `checkpoints/<experiment>/<mode>/<dataset>_<run>/`.

## Objectives

Available objectives are defined in `lamps/settings.py`:

| Objective | Description |
|---|---|
| `model/size_billion` | Model size, in billions of parameters |
| `eval/log_loss` | Validation cross-entropy (log scale) |
| `eval/neg_bleu` | Negative BLEU score (machine translation) |
| `eval/neg_accuracy` | Negative accuracy |
| `eval/neg_f1_macro` | Negative macro F1 |

Any combination can be passed via `--objectives` as long as the metrics are present in the
meta-dataset for the chosen experiment.

## Citation

```bibtex
@inproceedings{monteiro2026lamps,
  title     = {Landmark-Guided Policy Optimization for Multi-Objective Language Model Selection},
  author    = {Monteiro, M{\'a}rcio and Li, Weichen and Wang, Puyu and Kloft, Marius and Fellenz, Sophie},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {306},
  year      = {2026}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
