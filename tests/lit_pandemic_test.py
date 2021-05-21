from unittest import TestCase

from torch.nn.functional import mse_loss

from ode_nn import LitPandemic, Seiturd


class LitPandemitTest(TestCase):
    def test_smoke(self):
        """Smoke test of instantiation with a Seiturd model"""
        ds_shape = (365, 42, 50)  # days, populations, states
        model = Seiturd(ds_shape)
        lp = LitPandemic(model=model, loss_fn=mse_loss)
        assert lp is not None
        assert list(lp.parameters()) == list(model.parameters())
