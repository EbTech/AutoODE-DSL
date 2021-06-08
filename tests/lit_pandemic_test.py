from unittest import TestCase

from torch.nn.functional import mse_loss

from ode_nn import LitPandemic, SeiturdModel
from ode_nn.data import C19Dataset


class LitPandemitTest(TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a dataset and save an adjacency matrix."""
        cls.ds = C19Dataset()
        cls.A = cls.ds.adjacency

    def test_smoke(self):
        """Smoke test of instantiation with a Seiturd model"""
        model = SeiturdModel(num_days=len(self.ds), adjacency_matrix=self.A)
        lp = LitPandemic(model=model, loss_fn=mse_loss)
        assert lp is not None
        assert list(lp.parameters()) == list(model.parameters())
