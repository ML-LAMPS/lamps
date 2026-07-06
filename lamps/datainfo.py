import numpy as np

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

nds = NonDominatedSorting()


class DataInfo:

    def __init__(self, models: list[str], metrics: list[str], data: dict):

        rows = []

        for model in models:
            model_idx = models.index(model)
            model_data = data[model]
            num_epochs = len(model_data[metrics[0]])

            for epoch in range(num_epochs):
                row = [model_idx, epoch] + [
                    model_data[metric][epoch] for metric in metrics
                ]
                rows.append(row)

        self.data = np.array(rows)

    def model_data(self, model_idx: int, max_epoch: int | None = None):
        data = self.data[self.data[:, 0] == model_idx]
        if max_epoch is not None:
            return data[: max_epoch + 1, 2:]
        return data[:, 2:]

    def hypervolume(self, ref_point: list[float]):
        hv = HV(ref_point=ref_point)
        return hv.do(self.data[:, 2:])

    def best_epoch(self, model_idx: int, max_epoch: int | None = None):
        model_data = self.model_data(model_idx, max_epoch)
        front = nds.do(model_data, only_non_dominated_front=True)
        return front[0]
