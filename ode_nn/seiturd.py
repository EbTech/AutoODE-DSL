from __future__ import annotations

import random
from collections import namedtuple
from typing import List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils import data

from .data import C19Dataset


class State(NamedTuple):
    """
    The population state of a given Seiturd model.

    Each attribute should be size [n_regions].
    """

    S: torch.Tensor  # susceptible population
    E: torch.Tensor  # exposed population in the non-contagious incubation period
    I: torch.Tensor  # infected population (contagious but not tested)
    T: torch.Tensor  # infected population that has tested positive
    U: torch.Tensor  # undetected population, either self-quarantined or recovered
    R: torch.Tensor  # recovered population that has tested positive
    D: torch.Tensor  # death toll

    @property
    def N(self) -> torch.Tensor:
        """Total population"""
        return sum(self)

    def __add__(self, other: State) -> State:
        return State(*[a + b for (a, b) in zip(self, other)])

    def __sub__(self, other: State) -> State:
        return State(*[a - b for (a, b) in zip(self, other)])

    # As the transition model is a tree, flows are uniquely determined by delta,
    # since both have one fewer degree of freedom than the number of states.
    # To infer flows, proceed by eliminating one leaf at a time.
    def flows_to(self, other: State) -> Flows:
        delta = other - self

        T_D = delta.D
        T_R = delta.R
        I_U = delta.U
        I_T = delta.T + T_D + T_R  # other.T == self.T - T_D - T_R + I_T
        E_I = delta.I + I_T + I_U  # other.I == self.I - I_T - I_U + E_I
        S_E = delta.E + E_I  # other.E == self.E - E_I + S_E

        return Flows(S_E=S_E, E_I=E_I, I_T=I_T, I_U=I_U, T_R=T_R, T_D=T_D)


class Flows(NamedTuple):
    S_E: torch.Tensor
    E_I: torch.Tensor
    I_T: torch.Tensor
    I_U: torch.Tensor
    T_R: torch.Tensor
    T_D: torch.Tensor


class History:
    """
    Represents the values of the SEITURD subpopulations over a span of ``num_days``.

    This is computationally more efficient than creating a bunch of separate states.

    The S state is implicit (S = N - E - I - T - U - R - D); this choice *does*
    affect the gradient descent direction slightly, but seems unlikely
    that would really matter.

    The R state is also implicit (R = num_pos_and_alive - T).
    """

    # Let f(x,y) = F(x,y,1-x-y). Then,
    # df/dx = dF/dx - dF/dz,
    # df/dy = dF/dy - dF/dz.
    # Effectively, a gradient step does:
    # x/step += dF/dx - dF/dz,
    # y/step += dF/dy - dF/dz,
    # z/step += 2*dF/dz - dF/dx - dF/dy.
    #
    # Hmmm I suppose the asymmetry comes from the fact that we're using
    # (e_x-e_z, e_y-e_z) as our basis for the tangent space of the constraint
    # manifold (i.e., the plane x+y+z=1). If we want to interpret the gradient
    # as being the direction of greatest improvement, the basis should be
    # orthonormal! E.g., taking the vector (1,1,-2) and rotating +/-45 degrees
    # within the plane x+y+z=1 yields the orthogonal basis
    # ((a+1)e_x + (a-1)e_y, (a-1)e_x + (a+1)e_y), where a = 1/sqrt(3).
    # In higher dimensions, the rotation angle is arccos(1/sqrt(N-1)).

    fields = "SEITURD"  # could avoid hardcoding these, not bothering for now

    def __init__(
        self,
        N: torch.Tensor,  # total pop, shape [num_regions]
        num_pos_and_alive: torch.Tensor,  # T + R, shape [num_days, num_regions]
        num_dead: torch.Tensor,  # D, shape [num_days, num_regions]
        # num_days: int,
        requires_grad: bool = False,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.N = torch.as_tensor(N, device=device)
        self.num_pos_and_alive = torch.as_tensor(num_pos_and_alive, device=device)
        (self.num_days, self.num_regions) = self.num_pos_and_alive.shape
        assert self.N.shape == (self.num_regions,)
        self.num_dead = num_dead
        assert num_dead.shape == (self.num_days, self.num_regions)

        # 7 SEITURD states minus 1 population constraint minus 2 data-enforced
        # constraints equals 4 states to fit
        self.data = torch.full(
            (4, self.num_days, self.num_regions),
            0.0,
            device=device,
            requires_grad=requires_grad,
        )

    # S is implicit to make things sum to N (below)
    E = property(lambda self: self.data[0])
    I = property(lambda self: self.data[1])  # noqa: E741
    T = property(lambda self: self.data[2])
    U = property(lambda self: self.data[3])
    R = property(lambda self: self.num_pos_and_alive - self.T)  # implicit...
    D = property(lambda self: self.num_dead)

    @property
    def S(self):
        return (
            self.N.unsqueeze(0)
            - self.num_pos_and_alive
            - self.E
            - self.I
            - self.U
            - self.D
        )

    @classmethod
    def from_dataset(cls, dataset: C19Dataset, **kwargs):
        TRD = dataset.tensor[:, 0, :]
        D = dataset.tensor[:, -1, :]
        TR = TRD - D

        history = cls(N=dataset.pop_2018, num_pos_and_alive=TR, num_dead=D, **kwargs)
        return history

    def __getitem__(self, i: int):
        return State(*(getattr(self, n)[i] for n in self.fields))

    def __setitem__(self, i: int, state: State, check_consistency: bool = False):
        if check_consistency:
            assert torch.equal(state.N, self.N)
            assert torch.equal(state.T + state.R, self.num_pos_and_alive)
        for name, state_val in zip(State._fields, state):
            if name in "SR":
                continue
            getattr(self, name)[i] = state_val

    def __len__(self):
        return len(self.fields)

    def __repr__(self):
        return (
            f"<{type(self).__qualname__} object "
            f"({self.num_days} days, {self.num_regions} regions)>"
        )


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

    # run_one_step and run_forward are broken now, because we removed the
    # change functions....

    # def run_one_step(self, state: State, t: int) -> State:
    #     """
    #     Runs the model from ``history``; this model is Markov and
    #     therefore only looks at the *last* day of ``history``.
    #     """
    #     return state + State(
    #         *(getattr(self, f"change_{n}")(state, t) for n in State._fields)
    #     )
    #
    # def run_forward(
    #     self,
    #     initial_state: State,
    #     initial_t: int,
    #     num_days: int,
    #     requires_grad: bool = False,
    # ) -> History:
    #     """
    #     Walk the model forward from ``initial_state``.
    #
    #     Returns a new ``History`` object, covering days ``initial_t + 1``,
    #     ..., ``initial_t + num_days``.
    #
    #     (The new history uses the same device as ``initial_state``;
    #     ``requires_grad`` is passed along.)
    #     """
    #     future = History(
    #         self.num_regions,
    #         num_days,
    #         requires_grad=requires_grad,
    #         device=initial_state.S.device,
    #     )
    #
    #     curr_state = initial_state
    #     for i in range(num_days):
    #         curr_state = future[i] = self.run_one_step(curr_state, initial_t + i + 1)
    #     return future

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
    def prob_S_E(self, I_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        The fraction of ``S`` that is transformed into ``E``.

        Args:
          I_t (torch.Tensor): infected population ``I`` at time ``t``,
            of shape ``(num_regions,)``
          t (int): time index

        Returns:
          amount of ``S`` that changes into ``E``, of shape `(n_regions,)
        """
        # contagion_I is (n_days, num_regions)
        # adjacency_matrix is (n_regions, num_regions)
        # I_t must (num_regions,)
        return self.contagion_I[t] * (self.adjacency_matrix @ I_t)

    def prob_E_I(self) -> torch.Tensor:
        return self.decay_E

    def prob_I_out(self) -> torch.Tensor:  # I->T or I->U
        return self.decay_I

    def prob_I_T(self, t: int) -> torch.Tensor:
        return self.detection_rate[t] * self.prob_I_out()

    def prob_I_U(self, t: int) -> torch.Tensor:
        return (1 - self.detection_rate[t]) * self.prob_I_out()

    def prob_T_out(self) -> torch.Tensor:  # T->R or T->D
        return self.decay_T

    def prob_T_R(self, t: int) -> torch.Tensor:
        return self.recovery_rate[t] * self.prob_T_out()

    def prob_T_D(self, t: int) -> torch.Tensor:
        return (1.0 - self.recovery_rate[t]) * self.prob_T_out()

    # Distributions of population flows
    def flow_from_S(self, state: State, t: int) -> (torch.Tensor, torch.Tensor):
        p = [self.prob_S_E(state.I, t)]
        return flow_multinomial(state.S, p)

    def flow_from_E(self, state: State, t: int) -> (torch.Tensor, torch.Tensor):
        # t is only included to keep the interface uniform, it's not used
        p = [self.prob_E_I()]
        return flow_multinomial(state.E, p)

    def flow_from_I(self, state: State, t: int) -> (torch.Tensor, torch.Tensor):
        p = [self.prob_I_T(t), self.prob_I_U(t)]
        return flow_multinomial(state.I, p)

    def flow_from_T(self, state: State, t: int) -> (torch.Tensor, torch.Tensor):
        p = [self.prob_T_R(t), self.prob_T_D(t)]
        return flow_multinomial(state.T, p)


def flow_multinomial(n: int, p: torch.Tensor) -> (torch.Tensor, torch.Tensor):
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
    print(n.shape, p.shape)
    mean = n * p  # TODO - this line is busted
    cov = torch.diag_embed(mean) - n * p[:, :, np.newaxis] * p[:, np.newaxis, :]
    return mean, cov
