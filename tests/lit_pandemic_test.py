from unittest import TestCase

from torch.nn.functional import mse_loss

from ode_nn import LitPandemic, SeiturdModel


class LitPandemitTest(TestCase):
    def test_smoke(self):
        """Smoke test of instantiation with a Seiturd model"""
        model = SeiturdModel(365)
        lp = LitPandemic(model=model, loss_fn=mse_loss)
        assert lp is not None
        assert list(lp.parameters()) == list(model.parameters())
