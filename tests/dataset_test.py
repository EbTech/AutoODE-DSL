from unittest import TestCase

import torch

from ode_nn.data import C19Dataset


class C19DatasetTest(TestCase):
    def test_smoke(self):
        """Test the data instantiates"""
        ds = C19Dataset()
        assert ds is not None
        assert len(ds) > 0
        assert isinstance(ds[0], torch.Tensor)

    def test_adjacency(self):
        ds = C19Dataset()
        A = ds.get_adjacency()
        assert isinstance(A, torch.Tensor)
        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        n = len(ds.state_names)
        assert A.shape == (n, n)
        assert (A[0, :] == A[:, 0]).all()
