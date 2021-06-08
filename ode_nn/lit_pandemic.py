"""
Our lightning module will be responsible for implementing the training
and validation steps.
"""
from typing import Any, Callable, Dict, Tuple

import pytorch_lightning as pl
import torch


class LitPandemic(pl.LightningModule):
    """
    The pandemic lightning module is responsible for implementing
    training and validation steps. It abstracts away parts of the
    model that occur after the call to :meth:`forward`.

    Args:
      model (torch.nn.Module): the pandemic model. The only requirement
        is that it implements :meth:`forward`
      loss_fn (Callable): the loss function. It must have an API that takes in
        the population, the estimate, and the model itself
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor):
        """Aliases to :meth:`model.forward`. Unused."""
        # TODO - call run_forward properly
        return self.model.run_forward(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """
        The training step is responsible for computing a loss.

        .. todo:: implement this

        Args:
          batch (Tuple[torch.Tensor, ...]): contents of the batch
          batch_idx (int): ignored

        Returns:
          the loss
        """
        y_hat = self(batch)
        return self.loss_fn(batch, y_hat, self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    # A short unit test to make sure LitPandemic can instantiate
    class Dummy(torch.nn.Module):
        def forward(*args):
            return torch.rand(1)[0]  # return a random scalar

    def dummy_loss(*args):
        return torch.rand(1)[0]  # return a random scalar

    lp = LitPandemic(model=Dummy(), loss_fn=dummy_loss)
    assert lp is not None
    y_hat = lp(42)
    assert isinstance(y_hat, torch.Tensor)
    pars = list(lp.parameters())
    assert len(pars) == 0  # Dummy has no params
