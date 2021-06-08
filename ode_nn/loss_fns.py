"""
Loss functions designed according to the API in ``lit_pandemic.py``.
"""
from abc import ABC, abstractmethod

import torch

from ode_nn.seiturd import History


class LossFn(ABC):
    """
    Helpful base class meant to codify the API of our loss
    functions.
    """

    @abstractmethod
    def __call__(
        self, true_history: torch.Tensor, pred_history: History, model: torch.nn.Module
    ):
        pass  # pragma: no cover


class C19MSELoss(LossFn):
    mse = torch.nn.MSELoss()

    def __call__(
        self, true_history: torch.Tensor, pred_history: History, model: torch.nn.Module
    ):
        T = true_history[:, 0]
        D = true_history[:, 2]
        T_hat = pred_history.T
        D_hat = pred_history.D

        return self.mse(T, T_hat) + self.mse(D, D_hat)
