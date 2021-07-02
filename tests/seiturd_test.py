from unittest import TestCase

import pytest
import torch

from ode_nn.seiturd import SeiturdModel, State


class SeiturdTest(TestCase):
    def test_smoke(self):
        """Test that the model instantiates"""
        model = SeiturdModel(num_days=365)
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
        A = torch.rand((n_regions, n_regions))
        model = SeiturdModel(num_days=3, adjacency_matrix=A)
        pS = model.prob_S_E(I_t, t=0)
        assert isinstance(pS, torch.Tensor)
        assert pS.shape == I_t.shape

    def test_log_prob(self):
        assert False

    def test_flow_log_prob(Self):
        assert False
