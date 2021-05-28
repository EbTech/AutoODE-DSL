from unittest import TestCase

import pytest
import torch

from ode_nn.seiturd import Seiturd, State


class SeiturdTest(TestCase):
    def test_smoke(self):
        """Test that the model instantiates"""
        ds_shape = (365, 420, 50)  # days, populations, states
        model = Seiturd(ds_shape)
        assert model is not None

    def test_state(self):
        state = State(*torch.rand(7).reshape(-1, 1))
        for pop_name in list("SEITURD"):
            pop = getattr(state, pop_name)
            assert isinstance(pop, torch.Tensor)
            assert pop.shape == (1,)
        assert state.N == sum(state)
