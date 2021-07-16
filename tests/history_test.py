from unittest import TestCase

import numpy as np
import pytest
import torch

from ode_nn.data import C19Dataset
from ode_nn.history import HistoryWithImplicits, HistoryWithSoftmax, State


class HistoryTest(TestCase):
    def test_assign(self):
        ds = C19Dataset()
        for History in [HistoryWithSoftmax, HistoryWithImplicits]:
            with torch.no_grad():
                history = History.from_dataset(ds)
                logits = torch.softmax(torch.rand(7, history.num_regions, device=history.N.device), dim=0)
                SEITURD = logits * history.N[np.newaxis, :]
                state = State(*SEITURD)
                history[4] = state
                for source, exp in zip(history[4], state):
                    assert torch.allclose(source, exp)
