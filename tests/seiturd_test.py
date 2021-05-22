from unittest import TestCase

import pytest
import torch

from ode_nn import Seiturd


class SeiturdTest(TestCase):
    def test_smoke(self):
        """Test that the model instantiates"""
        ds_shape = (365, 42, 50)  # days, populations, states
        model = Seiturd(ds_shape)
        assert model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
    def test_device(self):
        """Test that the device is set"""
        ds_shape = (365, 42, 50)  # days, populations, states
        model = Seiturd(ds_shape, device=torch.device("cuda"))
        assert model.is_cuda
