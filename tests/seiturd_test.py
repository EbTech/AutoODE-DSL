from unittest import TestCase

import pytest
import torch

from ode_nn.data import C19Dataset
from ode_nn.seiturd import History, SeiturdModel, State, flow_multinomial


class FlowMultinomialTest(TestCase):
    def test_flow_multinomial(self):
        n = torch.Tensor([1, 3, 3])
        p = torch.Tensor([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
        mean, cov = flow_multinomial(n, p)
        assert mean.shape == (3, 2)
        assert cov.shape == (3, 2, 2)


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
        N = I_t + torch.rand(n_regions) * 10
        A = torch.rand((n_regions, n_regions))
        model = SeiturdModel(num_days=3, adjacency_matrix=A)
        pS = model.prob_S_E(I_t, N, t=0)
        assert isinstance(pS, torch.Tensor)
        assert pS.shape == I_t.shape

    def test_log_prob(self):
        ds = C19Dataset()
        A = ds.adjacency
        history = History.from_dataset(ds)
        model = SeiturdModel(history.num_days, A)
        history = History.from_dataset(ds)
        log_prob = model.log_prob(history)
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.dtype is torch.float
        assert log_prob.requires_grad  # means we can call log_prob.backward()
