"""
The ``SEITURD`` model, used to sample the joint posterior of populations
and physical parameters that model the spread of COVID-19.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Multinomial, MultivariateNormal

from ode_nn.history import Flows, History, State


class SeiturdModel(nn.Module):
    """
    A model for the Covid-19 pandemic given seven different populations
    in a compartment model:

    ``S`` - susceptible
    ``E`` - exposed
    ``I`` - infected
    ``T`` - tested
    ``U`` - undetected
    ``R`` - recovered
    ``D`` - deceased

    These populations are broken down so that ``T`` and ``D`` align with
    the data we can observe, i.e. the number of daily positive tests and
    the number of deceased people due to Covid-19 recorded daily.

    This class is constructed in a way to allow it to learn from a training
    set that is a tensor of shape `(num_days, n_populations, n_regions)`,
    where `num_days` is the number of days for which we have data,
    `n_populations` is the number of populations for which we have data for --
    either two (``T``, ``D``) or three (``T``, ``R``, ``D``) depending on which
    data is used. `n_regions` is the number of geographical regions under
    consideration

    .. note::

      the number of populations that contribute to the loss are
      defined in the  loss function used in the ``LitPandemic`` class.

    In order to *predict* populations for new dates... TODO

    .. todo: Update the init args

    Args:
      num_days (int): number of days to *train* on
      adjacency_matrix (torch.Tensor): a square tensor of shape
        `(n_regions, n_regions)`
    """

    def __init__(
        self,
        num_days: int,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ):
        assert num_days > 0
        super().__init__()

        self.adjacency_matrix = adjacency_matrix
        if adjacency_matrix is None:
            self.adjacency_matrix = torch.Tensor([[1]])

        assert self.adjacency_matrix.ndim == 2
        assert self.adjacency_matrix.shape[0] == self.adjacency_matrix.shape[1]

        num_regions = self.adjacency_matrix.shape[0]
        self.num_days = num_days
        self.num_regions = num_regions

        # TODO: determine what the typical scales are
        # TODO: initialize parameters to typical values & scales
        # lambda_E = rate of E -> I transition
        self.logit_decay_E = nn.Parameter(torch.logit(torch.rand([])))
        # lambda_I = rate of I -> {T, U} transition
        self.logit_decay_I = nn.Parameter(torch.logit(torch.rand([])))
        # lambda_T = rate of T -> {R, D} transition
        self.logit_decay_T = nn.Parameter(torch.logit(torch.rand([])))

        # d_i = detection rate
        self.detection_rate = nn.Parameter(torch.rand((num_days, num_regions)))
        # r_{i,t} = recovery rate
        self.recovery_rate = nn.Parameter(torch.rand((num_days, num_regions)))
        # beta_{i,t} = probability of infection per interaction with I person
        self.contagion_I = nn.Parameter(torch.rand((num_days, num_regions)))
        # eps_{i,t} = probability of infection per interaction with T person
        self.contagion_T = nn.Parameter(torch.rand((num_days, num_regions)))
        # A_{i,j} = percentage of infected people in state j who interact with
        # each susceptible person of state i
        self.connectivity = nn.Parameter(torch.eye(num_regions))

    @property
    def decay_E(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_decay_E)

    @property
    def decay_I(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_decay_I)

    @property
    def decay_T(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_decay_T)

    def log_prob(self, history: History) -> float:
        assert self.num_days == history.num_days

        total = 0
        for t in range(len(history) - 1):
            old = history[t]
            new = history[t + 1]
            flows = old.flows_to(new)
            total += self.flow_log_prob(flows, old, t)
        return total

    def flow_log_prob(self, flows: Flows, state: State, t: int) -> float:
        logp = 0

        # TODO: these and the flow_from_* functions should be made generic,
        # by keeping a dictionary or something of what the flows are
        S_dist = MultivariateNormal(*self.flow_from_S(state, t))
        logp += S_dist.log_prob(flows.S_E.unsqueeze(1))

        E_dist = MultivariateNormal(*self.flow_from_E(state, t))
        logp += E_dist.log_prob(flows.E_I.unsqueeze(1))

        I_dist = MultivariateNormal(*self.flow_from_I(state, t))
        logp += I_dist.log_prob(torch.stack((flows.I_T, flows.I_U), 1))

        T_dist = MultivariateNormal(*self.flow_from_T(state, t))
        logp += T_dist.log_prob(torch.stack((flows.T_R, flows.T_D), 1))

        return logp

    def sample_one_step(self, state: State, t: int) -> State:
        """
        Runs the model from ``history``; this model is Markov and
        therefore only looks at the *last* day of ``history``.
        """

        def do_sampling(ns, ps):
            """
            ps and the return value both have shape [num_regions, out_degree]
            """
            probs = torch.cat((ps, 1 - ps.sum(1, keepdim=True)), dim=1)
            dist = Multinomial(ns, probs=probs)
            return dist.sample().t()[:-1]

        (S_E,) = do_sampling(state.S, self.flow_from_S(state, t))
        (E_I,) = do_sampling(state.E, self.flow_from_E(state, t))
        I_T, I_U = do_sampling(state.I, self.from_from_I(state, t))
        T_R, T_D = do_sampling(state.T, self.from_from_T(state, t))

        flows = Flows(S_E=S_E, E_I=E_I, I_T=I_T, I_U=I_U, T_R=T_R, T_D=T_D)
        return state.add_flow(flows)

    def sample_forward(
        self,
        initial_state: State,
        initial_t: int,
        num_days: int,
        requires_grad: bool = False,
    ) -> History:
        """
        Sample the model forward from ``initial_state``.

        Returns a new ``History`` object, covering days ``initial_t + 1``,
        ..., ``initial_t + num_days``.

        (The new history uses the same device as ``initial_state``;
        ``requires_grad`` is passed along.)
        """
        future = History(
            self.num_regions,
            num_days,
            requires_grad=requires_grad,
            device=initial_state.S.device,
        )

        curr_state = initial_state
        for i in range(num_days):
            curr_state = future[i] = self.sample_one_step(curr_state, initial_t + i + 1)
        return future

    # The states are Markov; that is, transitions depend only on the current
    # state. prob_X_Y gives the probability for a member of the population at
    # state X, to transition into state Y.
    # The transition graph is:
    #           /->D
    # S->E->I->T->R
    #        \->U
    # Note: I contributes to (U,T) and T contributes to (R,D)

    # NOTE: We are currently assuming that people in the T state are not
    # contagious, i.e., eps_{i,t} = 0.
    def prob_S_E(self, I_t: torch.Tensor, N: torch.Tensor, t: int) -> torch.Tensor:
        """
        The fraction of ``S`` that is transformed into ``E``.

        Args:
          I_t (torch.Tensor): infected population ``I`` at time ``t``,
            of shape ``(num_regions,)``
          N (torch.Tensor): total population in each region
            (shape ``[num_regions]``)
          t (int): time index

        Returns:
          amount of ``S`` that changes into ``E``, of shape `(n_regions,)
        """
        # contagion_I is (n_days, num_regions)
        # adjacency_matrix is (n_regions, num_regions)

        infectious_pops = self.contagion_I[t] * (self.adjacency_matrix @ I_t)
        total_pops = self.adjacency_matrix @ N
        return -torch.expm1(-infectious_pops / total_pops)

    # These prob_* functions return "something that can broadcast with an
    # array of shape [num_regions]": prob_S_E actually varies per region
    # and so returns shape [num_regions], but the others currently don't and
    # so return shape [1].

    def prob_E_I(self) -> torch.Tensor:
        return self.decay_E.unsqueeze(0)

    def prob_I_out(self) -> torch.Tensor:  # I->T or I->U
        return self.decay_I.unsqueeze(0)

    def prob_I_T(self, t: int) -> torch.Tensor:
        return self.detection_rate[t] * self.prob_I_out()

    def prob_I_U(self, t: int) -> torch.Tensor:
        return (1 - self.detection_rate[t]) * self.prob_I_out()

    def prob_T_out(self) -> torch.Tensor:  # T->R or T->D
        return self.decay_T.unsqueeze(0)

    def prob_T_R(self, t: int) -> torch.Tensor:
        return self.recovery_rate[t] * self.prob_T_out()

    def prob_T_D(self, t: int) -> torch.Tensor:
        return (1.0 - self.recovery_rate[t]) * self.prob_T_out()

    # Distributions of population flows
    def flow_from_S(self, state: State, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.prob_S_E(state.I, state.N, t).unsqueeze(1)
        return flow_multinomial(state.S, p)

    def flow_from_E(self, state: State, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # t is only included to keep the interface uniform, it's not used
        p = self.prob_E_I().unsqueeze(1)
        return flow_multinomial(state.E, p)

    def flow_from_I(self, state: State, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = torch.stack([self.prob_I_T(t), self.prob_I_U(t)], dim=1)
        return flow_multinomial(state.I, p)

    def flow_from_T(self, state: State, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = torch.stack([self.prob_T_R(t), self.prob_T_D(t)], dim=1)
        return flow_multinomial(state.T, p)


def flow_multinomial(
    n: torch.Tensor, p: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes mean and covariance of a multinomial.

    Arguments:
      - n, number of trials, of shape [n_regions]
      - p, probabilities, of shape [n_regions, n_outs].
        p should be nonnegative, and sum to at most 1.
    Returns:
      - mean of shape [n_regions, n_outs]
      - cov of shape [n_regions, n_outs, n_outs]
    """
    mean = n.unsqueeze(1) * p
    p_outer = p.unsqueeze(2) * p.unsqueeze(1)
    cov = torch.diag_embed(mean) - n[:, np.newaxis, np.newaxis] * p_outer
    return mean, cov
