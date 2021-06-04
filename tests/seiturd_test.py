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

    def test_one_step(self):
        n_regions = 50
        A = torch.rand((n_regions, n_regions))
        state = State(*torch.rand(7, n_regions))
        model = SeiturdModel(365, A)
        t: int = 0
        new_state = model.run_one_step(state, t)
        for pop_name in list("SEITURD"):
            # Check that there was a change
            assert (getattr(new_state, pop_name) != getattr(state, pop_name)).all()

    def test_run_forward(self):
        n_regions = 50
        A = torch.rand((n_regions, n_regions))
        state = State(*torch.rand(7, n_regions))
        model = SeiturdModel(365, A)
        n_days = 10
        hist = model.run_forward(state, 0, n_days)
        assert hist.S.shape == (n_days, n_regions)
