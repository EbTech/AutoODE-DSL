from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from ode_nn.data import C19Dataset


class C19DatasetTest(TestCase):
    def setUp(self):
        self.ds = C19Dataset()  # is stateless so this is OK

    def test_smoke(self):
        """Test the data instantiates"""
        ds = self.ds
        assert ds is not None
        assert len(ds) > 0
        assert isinstance(ds[0], torch.Tensor)

    def test_adjacency(self):
        ds = self.ds
        A = ds.get_adjacency()
        assert isinstance(A, torch.Tensor)
        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        n = len(ds.state_names)
        assert A.shape == (n, n)
        assert (A[0, :] == A[:, 0]).all()

    def test_in_dataloader(self):
        ds = self.ds
        dst = ds.tensor
        dl = DataLoader(dataset=ds, batch_size=len(ds))
        batch = next(iter(dl))
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == dst.shape
        assert torch.allclose(batch, dst, equal_nan=True)
