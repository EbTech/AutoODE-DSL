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
