from __future__ import annotations

import random
from collections import namedtuple
from typing import List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class State(NamedTuple):
    """
    The population state of a given Seiturd model.
    """

    S: torch.Tensor
    E: torch.Tensor
    I: torch.Tensor
    T: torch.Tensor
    U: torch.Tensor
    R: torch.Tensor
    D: torch.Tensor

    @property
    def N(self) -> torch.Tensor:
        """Total population"""
        return sum(self)
        # return self.S + self.E + self.I + self.T + self.U + self.R + self.D

    def __add__(self, oth: State) -> State:
        return State(*[a + b for (a, b) in zip(self, oth)])


class History:
    """
    Represents the values of the SEITURD subpopulations over a span of ``num_days``.

    A Markov state should be represented as a ``History`` object with ``num_days=1``.
    This is computationally more efficient than creating a bunch of separate states.
    """

    def __init__(
        self,
        num_regions: int,
        num_days: int,
        requires_grad: bool = False,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        shp = (num_days, num_regions)

        # S = susceptible population
        self.S = torch.full(shp, 0.5, device=device, requires_grad=requires_grad)

        # E = exposed population in the non-contagious incubation period
        self.E = torch.full(shp, 0.5, device=device, requires_grad=requires_grad)

        # I = infected population (contagious but not tested)
        self.I = torch.full(  # noqa: E741
            shp, 0.5, device=device, requires_grad=requires_grad
        )

        # T = infected population that has tested positive
        self.T = torch.full(shp, 0.5, device=device, requires_grad=requires_grad)

        # U = undetected population that has either self-quarantined or recovered
        self.U = torch.full(shp, 0.5, device=device, requires_grad=requires_grad)

        # R = recovered population that has tested positive
        self.R = torch.full(shp, 0.5, device=device, requires_grad=requires_grad)

        # D = death toll
        self.D = torch.full(shp, 0.5, device=device, requires_grad=requires_grad)

    def __getitem__(self, i):
        return State(*(getattr(self, n)[i] for n in State._fields))

    def __setitem__(self, i, state):
        for n in State._fields:
            getattr(self, n)[i] = getattr(state, n)


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
        self.ln_decay_E = nn.Parameter(torch.log(torch.rand(1)))
        # lambda_I = rate of I -> {T, U} transition
        self.ln_decay_I = nn.Parameter(torch.log(torch.rand(1)))
        # lambda_T = rate of T -> {R, D} transition
        self.ln_decay_T = nn.Parameter(torch.log(torch.rand(1)))

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
        return torch.exp(self.ln_decay_E)

    @property
    def decay_I(self) -> torch.Tensor:
        return torch.exp(self.ln_decay_I)

    @property
    def decay_T(self) -> torch.Tensor:
        return torch.exp(self.ln_decay_T)

    def run_one_step(self, state: State, t: int) -> State:
        """
        Runs the model from ``history``; this model is Markov and
        therefore only looks at the *last* day of ``history``.
        """
        return state + State(
            *(getattr(self, f"change_{n}")(state, t) for n in State._fields)
        )

    def run_forward(
        self,
        initial_state: State,
        initial_t: int,
        num_days: int,
        requires_grad: bool = False,
    ) -> History:
        """
        Walk the model forward from ``initial_state``.

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
            curr_state = future[i] = self.run_one_step(curr_state, initial_t + i + 1)
        return future

    # The states are Markov; that is, transitions depend only on the current
    # state. prob_X_Y gives the probability for a member of the population at
    # state X, to transition into state Y.
    # The transition graph is:
    # S->E->I->T->R
    #        \->U  \->D
    # Note: I contributes to (U,T) and T contributes to (R,D)

    # NOTE: We are currently assuming that people in the T state are not
    # contagious, i.e., eps_{i,t} = 0.
    def prob_S_E(self, I_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        The fraction of ``S`` that is transformed into ``E``.

        Args:
          I_t (torch.Tensor): infected population ``I`` at time ``t``,
            of shape ``(num_regions,)``
          t (int): time index

        Returns:
          amound of ``S`` that changes into ``E``, of shape `(n_regions,)
        """
        # contagion_I is (n_days, num_regions)
        # adjacency_matrix is (n_regions, num_regions)
        # I_t must (num_regions,)
        return self.contagion_I[t] * (self.adjacency_matrix @ I_t)

    def prob_E_I(self):
        return self.decay_E

    def prob_I_out(self):  # I->T or I->U
        return self.decay_I

    def prob_I_T(self):
        return self.detection_rate * self.prob_I_out()

    def prob_I_U(self):
        return (1 - self.detection_rate) * self.prob_I_out()

    def prob_T_out(self):  # T->R or T->D
        return self.decay_T

    def prob_T_R(self):
        return self.recovery_rate * self.prob_T_out()

    def prob_T_D(self):
        return (1.0 - self.recovery_rate) * self.prob_T_out()

    # Net population change
    def change_S(self, state, t):
        return -self.prob_S_E(state.I, t) * state.S

    def change_E(self, state, t):
        return self.prob_S_E(state.I, t) * state.S - self.prob_E_I() * state.E

    def change_I(self, state, t):  # ignores t
        return self.prob_E_I() * state.E - self.prob_I_out() * state.I

    def change_T(self, state, t):  # ignores t
        return self.prob_I_T() * state.I - self.prob_T_out() * state.T

    def change_U(self, state, t):  # ignores t
        return self.prob_I_U() * state.I

    def change_R(self, state, t):  # ignores t
        return self.prob_T_R() * state.T

    def change_D(self, state, t):  # ignores t
        return self.prob_T_D() * state.T
