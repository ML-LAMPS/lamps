import json
import os
import re

from glob import glob
from functools import lru_cache

import numpy as np

from tbparse import SummaryReader

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from lamps.datainfo import DataInfo

nds = NonDominatedSorting()


def parse_model_name(directory: str):
    """
    Parse the model name from a given directory string.
    """
    m = re.search(r"(?<=\bm=).+?(?=__)", directory)
    return m.group(0).replace("_SL_", "/")


@lru_cache(maxsize=None)
def parse_tblogs(log_dir: str):

    with open("model_sizes.json", "r", encoding="utf-8") as f:
        model_sizes = json.load(f)

    try:
        reader = SummaryReader(log_dir)
        df = reader.scalars

        eval_data = {
            k: df[df.tag == k]["value"].tolist()
            for k in df.tag.unique()
            if k.startswith("eval/")
        }
    except (FileNotFoundError, ValueError):
        eval_data = {
            "eval/loss": [],
            "eval/epoch_runtime": [],
        }
        print(f"Warning: No logs found for {log_dir}")

    # model/size
    assert (
        parse_model_name(log_dir) in model_sizes
    ), "Model size not found for model_sizes.json"

    eval_data["model/size"] = [model_sizes[parse_model_name(log_dir)]] * len(
        eval_data["eval/loss"]
    )
    # model/size_billion
    eval_data["model/size_billion"] = [size / 1e9 for size in eval_data["model/size"]]
    # eval/log_loss
    eval_data["eval/log_loss"] = [np.log(loss) for loss in eval_data["eval/loss"]]

    if "eval/bleu" in eval_data:
        eval_data["eval/neg_bleu"] = [-bleu for bleu in eval_data["eval/bleu"]]

    return eval_data



class Repository:

    experiment: str
    dataset: str
    data: dict

    @property
    def models(self):
        _models = list(self.data.keys())
        _models.sort()

        return _models

    @property
    def num_models(self):
        return len(self.data.keys())

    def __init__(self, experiment: str, dataset: str, metrics: list[str]):
        self.experiment = experiment
        self.dataset = dataset
        self.metrics = metrics

        assert dataset in self.list_datasets(
            experiment
        ), f"Dataset {dataset} not found in experiment {experiment}."

        self.load_data()
        if not self.data:
            raise ValueError(
                f"No model data found for experiment={experiment}, dataset={dataset}. "
                "Verify metadataset paths are populated."
            )
        if not self.models:
            raise ValueError(
                f"No models available for experiment={experiment}. "
                "Repository contains no entries under metadataset/"
            )
        self.datainfo = DataInfo(self.models, metrics, self.data)

    def load_data(self, lr=5e-5, bs="auto"):
        data = {}

        m = re.search(r"^([^\[\]]+)(?:\[(.*?)\])?$", self.dataset)

        dataset_name = m.group(1)
        dataset_config = m.group(2) if m.group(2) else "none"
        models = self.list_models(self.experiment)

        for model in models:
            log_dir = f"metadataset/{self.experiment}/m={model.replace('/', '_SL_')}__d={dataset_name.replace('/', '_SL_')}__s={dataset_config}__lr={lr}__bs={bs}/"
            data[model] = parse_tblogs(log_dir)

        self.data = data

        return data

    def get_metric(self, model: str, metric: str, epoch: int):
        if model not in self.data:
            raise ValueError(f"Model {model} not found in data.")

        if metric not in self.data[model]:
            available_options = list(self.data[model].keys())
            raise ValueError(
                f"Metric {metric} not found for model {model}. Available options are: {available_options}"
            )

        if epoch >= len(self.data[model][metric]):
            raise ValueError(
                f"Epoch {epoch} out of range for metric {metric} of model {model}."
            )

        return self.data[model][metric][epoch]

    def get_num_available_epochs(self, model: str):
        return len(self.data[model]["eval/loss"]) - 1  # Exclude epoch 0

    def get_elapsed_time(self, model: str, epoch: int):
        return sum(self.data[model]["eval/epoch_runtime"][0 : epoch + 1])

    def get_total_time(self, only_pareto: bool = False):
        """
        Get the total training time to fine-tune all models for the dataset.
        """
        total_time = 0.0

        if only_pareto:
            models = self.get_pareto_models()

            # The epoch 0 runtime should be considered when evaluating total time,
            # even if the model is not Pareto-optimal.
            for model in self.models:
                if model in models:
                    continue
                total_time += self.data[model]["eval/epoch_runtime"][0]
        else:
            models = self.models

        for model in models:
            total_time += sum(self.data[model]["eval/epoch_runtime"])

        return total_time

    def get_optimal_epochs(self):
        optimal_epochs = np.array(
            [
                self.datainfo.best_epoch(self.models.index(model))
                for model in self.models
            ]
        )
        return optimal_epochs

    def get_available_epochs(self):
        available_epochs = np.array(
            [self.get_num_available_epochs(model) for model in self.models]
        )
        return available_epochs

    def get_pareto_models(self, exclude_models=()):
        """
        Get the models that are part of the Pareto front for the given objectives.
        """
        models = [model for model in self.models if model not in exclude_models]
        datapoints = self.datapoints(
            self.get_optimal_epochs(), exclude_models=exclude_models
        )
        front_indices = nds.do(datapoints, only_non_dominated_front=True)
        pareto_models = [models[i] for i in front_indices]
        pareto_models.sort()
        return pareto_models

    def get_optimal_hv(self, ref_point: list):
        datapoints = self.datapoints(self.get_optimal_epochs())

        if datapoints.size == 0:
            raise ValueError(
                "Cannot compute optimal hypervolume: repository has no datapoints."
            )

        hv = HV(ref_point=ref_point)
        hv_values = hv.do(datapoints)

        return hv_values

    def datapoints(
        self,
        epoch_counts: np.ndarray,
        preserve_best: bool = False,
        exclude_models: tuple = (),
    ):
        """
        Get the data points for the given objectives.
        """
        data_points = []
        for model_idx, model in enumerate(self.models):
            if model in exclude_models:
                continue
            current_epoch = epoch_counts[model_idx]
            best_epoch = self.datainfo.best_epoch(model_idx, current_epoch)

            _epoch = best_epoch if preserve_best else current_epoch
            values = [self.data[model][obj][_epoch] for obj in self.metrics]
            data_points.append(values)

        data_points = np.array(data_points)

        return data_points

    @staticmethod
    def list_models(experiment: str):
        """
        List all models in the repository for a given experiment.
        """
        base_dir = f"metadataset/{experiment}"
        models = []

        for directory in glob(f"{base_dir}/*"):
            _model = parse_model_name(directory)
            models.append(_model)

        models = list(set(models))
        models.sort()

        return models

    @staticmethod
    def list_datasets(experiment: str):
        """
        List all datasets in the repository for a given experiment.
        """
        base_dir = f"metadataset/{experiment}"
        datasets = []

        for directory in glob(f"{base_dir}/*"):
            ds = re.search(r"__d=(.+?)(?=__s=)", directory)
            cf = re.search(r"(?:^|__)s=((?:(?!__).)*)", directory)

            ds_name = ds.group(1).replace("_SL_", "/")
            ds_config = cf.group(1)

            if ds_config != "none":
                datasets.append(f"{ds_name}[{ds_config}]")
            else:
                datasets.append(ds_name)

        datasets = list(set(datasets))
        datasets.sort()

        return datasets


