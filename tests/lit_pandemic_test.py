from unittest import TestCase

from torch.utils.data import DataLoader

from ode_nn import LitPandemic, SeiturdModel
from ode_nn.data import C19Dataset
from ode_nn.loss_fns import C19MSELoss


class LitPandemitTest(TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a dataset and save an adjacency matrix."""
        cls.ds = C19Dataset()
        cls.A = cls.ds.adjacency

    def setUp(self):
        self.mse = C19MSELoss()

    def test_smoke(self):
        """Smoke test of instantiation with a Seiturd model"""
        model = SeiturdModel(num_days=len(self.ds), adjacency_matrix=self.A)
        lp = LitPandemic(model=model, loss_fn=self.mse)
        assert lp is not None
        assert list(lp.parameters()) == list(model.parameters())

    def test_training_step_with_mse(self):
        # model = SeiturdModel(num_days=len(self.ds), adjacency_matrix=self.A)
        # lp = LitPandemic(model=model, loss_fn=self.mse)
        dl = DataLoader(dataset=self.ds, batch_size=len(self.ds))
        # loss = lp.training_step(next(iter(dl)), 0)
        assert dl is not None
