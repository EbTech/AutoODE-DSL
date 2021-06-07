"""
Loss functions designed according to the API in ``lit_pandemic.py``.
"""
from abc import ABC, abstractmethod

import torch


class LossFn(ABC):
    """
    Helpful base class meant to codify the API of our loss
    functions.
    """

    @abstractmethod
    def __call__(self, batch, y_hat: torch.tensor, model: torch.nn.Module):
        pass  # pragma: no cover
