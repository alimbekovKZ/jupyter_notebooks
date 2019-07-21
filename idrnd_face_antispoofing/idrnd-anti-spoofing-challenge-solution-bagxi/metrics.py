# Copyright (c) 2019-present, Yauheni Kachan. All Rights Reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

from catalyst.dl.core import Callback, RunnerState
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


class BinaryAUCCallback(Callback):
    def __init__(
        self,
        input_key: str = 'targets',
        output_key: str = 'logits',
        prefix: str = 'auc',
    ):
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key

        self.y_true = []
        self.y_proba = []

    def _reset_stats(self):
        self.y_true = []
        self.y_proba = []

    def on_loader_start(self, state: RunnerState):
        self._reset_stats()

    def on_batch_end(self, state: RunnerState):
        logits: torch.Tensor = state.output[self.output_key].detach().float()
        targets: torch.Tensor = state.input[self.input_key].detach().float()
        probabilities: torch.Tensor = torch.sigmoid(logits)

        self.y_true.append(targets.cpu().data.numpy())
        self.y_proba.append(probabilities.cpu().data.numpy()[:, 1])

    def on_loader_end(self, state: RunnerState):
        y_true = np.concatenate(self.y_true)
        y_score = np.concatenate(self.y_proba)
        area = float(roc_auc_score(y_true, y_score))
        state.metrics.epoch_values[state.loader_name][f'{self.prefix}'] = area

        self._reset_stats()
