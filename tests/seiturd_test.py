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

    def test_prob_S_E(self):
        n_regions = 50
        I_t = torch.rand(n_regions)
        model = Seiturd((42, 69, n_regions))
        # TODO: replace this 'fake' adjacency matrix with something real
        model.A = torch.rand((n_regions, n_regions))
        pS = model.prob_S_E(I_t, t=0)
        assert isinstance(pS, torch.Tensor)
        assert pS.shape == I_t.shape

    def test_one_step(self):
        n_regions = 50
        state = State(*torch.rand(7, n_regions))
        ds_shape = (365, 420, n_regions)  # days, populations, states
        model = Seiturd(ds_shape)
        # TODO: replace this 'fake' adjacency matrix with something real
        model.A = torch.rand((n_regions, n_regions))
        t: int = 0
        new_state = model.one_step(t, state)
        for pop_name in list("SEITURD"):
            # Check that there was a change
            assert (getattr(new_state, pop_name) != getattr(state, pop_name)).all()
