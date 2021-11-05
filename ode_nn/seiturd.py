"""
The ``SEITURD`` model, used to sample the joint posterior of populations
and physical parameters that model the spread of COVID-19.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Multinomial

from .history import Flows, History, State
from .linalg_utils import BivariateNormal, UnivariateNormal


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

        if adjacency_matrix is None:
            adjacency_matrix = torch.eye(1)
        assert adjacency_matrix.ndim == 2
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        self.num_days = num_days
        self.num_regions = num_regions = self.adjacency_matrix.shape[0]

        # TODO: determine what the typical scales are
        # TODO: initialize parameters to typical values & scales
        # lambda_E = rate of E -> I transition
        self.logit_decay_E = nn.Parameter(torch.logit(torch.rand([])))
        # lambda_I = rate of I -> {T, U} transition
        self.logit_decay_I = nn.Parameter(torch.logit(torch.rand([])))
        # lambda_T = rate of T -> {R, D} transition
        self.logit_decay_T = nn.Parameter(torch.logit(torch.rand([])))

        # d_i = detection rate
        self.logit_detection_rate = nn.Parameter(
            torch.logit(torch.rand((num_days, num_regions)))
        )
        # r_{i,t} = recovery rate
        self.logit_recovery_rate = nn.Parameter(
            torch.logit(torch.rand((num_days, num_regions)))
        )
        # beta_{i,t} = number of potentially-contagious interactions per day
        self.log_contagion_I = nn.Parameter(
            torch.log(torch.rand((num_days, num_regions)))
        )
        # eps_{i,t} = number of potentially-contagious interactions with T
        #             people per day; should be low if T people stay home,
        #             so for now just removing from model and clamping to 0
        # self.logit_contagion_T = nn.Parameter(
        #    torch.logit(torch.rand((num_days, num_regions)))
        # )
        # A_{i,j} = relative frequency of interaction across regions
        #           -- for now, we're just using self.adjacency_matrix
        # self.connectivity = nn.Parameter(torch.eye(num_regions))

    decay_E = property(lambda self: torch.sigmoid(self.logit_decay_E))
    decay_I = property(lambda self: torch.sigmoid(self.logit_decay_I))
    decay_T = property(lambda self: torch.sigmoid(self.logit_decay_T))
    detection_rate = property(lambda self: torch.sigmoid(self.logit_detection_rate))
    recovery_rate = property(lambda self: torch.sigmoid(self.logit_recovery_rate))
    contagion_I = property(lambda self: torch.exp(self.log_contagion_I))
    # contagion_T = property(lambda self: torch.exp(self.log_contagion_T))

    def log_prob(self, history: History) -> torch.Tensor:
        assert self.num_days == history.num_days

        total = 0
        for t in range(len(history) - 1):
            old = history[t]
            new = history[t + 1]
            flows = old.flows_to(new)
            total += self.flow_log_prob(flows, old, t)
        return total.mean() / (len(history) - 1)

    def flow_log_prob(self, flows: Flows, state: State, t: int) -> torch.Tensor:
        logp = 0

        # TODO: these and the flow_from_* functions should be made generic,
        # by keeping a dictionary or something of what the flows are
        S_dist = UnivariateNormal(*self.flow_from("S", state, t))
        logp += S_dist.log_prob(flows.S_E.unsqueeze(1))

        E_dist = UnivariateNormal(*self.flow_from("E", state, t))
        logp += E_dist.log_prob(flows.E_I.unsqueeze(1))

        I_dist = BivariateNormal(*self.flow_from("I", state, t))
        logp += I_dist.log_prob(torch.stack((flows.I_T, flows.I_U), 1))

        T_dist = BivariateNormal(*self.flow_from("T", state, t))
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

        (S_E,) = do_sampling(state.S, self.probs_from_S(state, t))
        (E_I,) = do_sampling(state.E, self.probs_from_E(state, t))
        I_T, I_U = do_sampling(state.I, self.probs_from_I(state, t))
        T_R, T_D = do_sampling(state.T, self.probs_from_T(state, t))

        flows = Flows(S_E=S_E, E_I=E_I, I_T=I_T, I_U=I_U, T_R=T_R, T_D=T_D)
        return state.add_flow(flows)

    def sample_forward(
        self,
        initial_state: State,
        initial_t: int,
        num_days: int,
    ) -> History:
        """
        Sample the model forward from ``initial_state``.

        Returns a new ``History`` object, covering days ``initial_t + 1``,
        ..., ``initial_t + num_days``.

        (The new history uses the same device as ``initial_state``.)
        """
        states = [initial_state]
        for i in range(num_days):
            states.append(self.sample_one_step(states[-1], initial_t + i + 1))
        return History.from_states(states)

    def many_samples(
        self,
        initial_state: State,
        initial_t: int,
        days: list[int],
        num_samples: int,
    ) -> torch.Tensor:
        "Returns a tensor with axes [day, region, SEITURD, sample_idx]."
        max_day = max(days)
        samps = np.empty(
            (len(days), self.num_regions, len(History.fields), num_samples)
        )
        for sample_id in range(num_samples):
            history = self.sample_forward(initial_state, initial_t, max_day)
            for day_id, day in enumerate(days):
                state = history[day]
                for field_id, field in enumerate(History.fields):
                    samps[day_id, :, field_id, sample_id] = getattr(state, field)
        return samps

    def sample_quantiles(
        self,
        initial_state: State,
        initial_t: int,
        days: list[int],
        quantiles: list[int],
        num_samples: int,
    ) -> torch.Tensor:
        "Returns a tensor with axes [day, region, SEITURD, quantile]."
        samps = self.many_samples(
            initial_state,
            initial_t,
            days,
            num_samples=num_samples,
        )
        return torch.quantile(samps, quantiles, dim=3)

    # The states are Markov; that is, transitions depend only on the current
    # state. prob_X_Y gives the probability for a member of the population at
    # state X, to transition into state Y.
    # The transition graph is:
    #           /->D
    # S->E->I->T->R
    #        \->U
    # Note: I contributes to (U,T) and T contributes to (R,D)

    # These prob_* functions return "something that can broadcast with an
    # array of shape [num_regions]": prob_S_E actually varies per region
    # and so returns shape [num_regions], but the others currently don't and
    # so return shape [1].

    # NOTE: We are currently assuming that people in the T state are not
    # contagious, i.e., eps_{i,t} = 0.
    def probs_from_S(self, state: State, t: int) -> torch.Tensor:
        # S to E
        # contagion_I is (n_days, num_regions)
        # adjacency_matrix is (n_regions, num_regions)

        infectious_pops = self.contagion_I[t] * (self.adjacency_matrix @ state.I)
        total_pops = self.adjacency_matrix @ state.N
        return -torch.expm1(-infectious_pops / total_pops).unsqueeze(1)

    def probs_from_E(self, state: State, t: int) -> torch.Tensor:
        # E to I
        return self.decay_E.unsqueeze(0).unsqueeze(1)

    def probs_from_I(self, state: State, t: int) -> torch.Tensor:
        # I to T,U
        prob_I_out = self.decay_I.unsqueeze(0).unsqueeze(1)  # [1, 1]
        detection_rate = self.detection_rate[t]  # [num_regions]
        return prob_I_out * torch.stack([detection_rate, 1 - detection_rate], dim=1)

    def probs_from_T(self, state: State, t: int) -> torch.Tensor:
        # T to R,D
        prob_T_out = self.decay_T.unsqueeze(0).unsqueeze(1)
        recovery_rate = self.recovery_rate[t]  # [num_regions]
        return prob_T_out * torch.stack([recovery_rate, 1 - recovery_rate], dim=1)

    _probs_from_fn = {
        "S": probs_from_S,
        "E": probs_from_E,
        "I": probs_from_I,
        "T": probs_from_T,
    }

    def probs_from(self, compartment: str, state: State, t: int):
        return self._probs_from_fn[compartment](self, state, t)

    def flow_from(
        self, compartment: str, state: State, t: int, fudge: float = 1e-7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return flow_multinomial(
            getattr(state, compartment),
            self.probs_from(compartment, state, t),
            fudge=fudge,
        )


def flow_multinomial(
    n: torch.Tensor, p: torch.Tensor, fudge: float = 1e-7
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
    cov = (
        cov
        + fudge * torch.eye(cov.shape[1], out=torch.empty_like(cov))[np.newaxis, :, :]
    )  # blegh
    return mean, cov
