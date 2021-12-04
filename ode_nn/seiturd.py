"""
The ``SEITURD`` model, used to sample the joint posterior of populations
and physical parameters that model the spread of COVID-19.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple
import warnings

import numpy as np
import torch
from torch import nn
from torch.distributions import Dirichlet

from .history import Flows, History, State, FLOWS_GRAPH
from .linalg_utils import BivariateNormal, TrivariateNormal, UnivariateNormal


class TransformedTensor:
    def __init__(self, base, func):
        self.base = base
        self.func = func

    def __getitem__(self, *args):
        return self.func(self.base[args])

    def evaluate(self):
        # is there a magic method for this?
        return self.func(self.base)

    def __getattr__(self, name):
        return getattr(self.evaluate(), name)


SigmoidTensor = partial(TransformedTensor, func=torch.sigmoid)


class BaseSeiturdModel(nn.Module):
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
        self.decay_E = SigmoidTensor(self.logit_decay_E)

        # lambda_I = rate of I -> {T, U} transition
        self.logit_decay_I = nn.Parameter(torch.logit(torch.rand([])))
        self.decay_I = SigmoidTensor(self.logit_decay_I)

        # lambda_T = rate of T -> {R, D} transition
        self.logit_decay_T = nn.Parameter(torch.logit(torch.rand([])))
        self.decay_T = SigmoidTensor(self.logit_decay_T)

        # d_i = detection rate
        self.logit_detection_rate = nn.Parameter(
            torch.logit(torch.rand((num_days, num_regions)))
        )
        self.detection_rate = SigmoidTensor(self.logit_detection_rate)

        # r_{i,t} = recovery rate
        self.logit_recovery_rate = nn.Parameter(
            torch.logit(torch.rand((num_days, num_regions)))
        )
        self.recovery_rate = SigmoidTensor(self.logit_recovery_rate)

        # beta_{i,t} = number of potentially-contagious interactions per day
        self.log_contagion_I = nn.Parameter(
            torch.log(torch.rand((num_days, num_regions)))
        )
        self.contagion_I = TransformedTensor(self.log_contagion_I, torch.exp)

        # eps_{i,t} = number of potentially-contagious interactions with T
        #             people per day; should be low if T people stay home,
        #             so for now just removing from model and clamping to 0
        # self.log_contagion_T = nn.Parameter(
        #    torch.log(torch.rand((num_days, num_regions)))
        # )
        # self.contagion_T = TransformedTensor(self.log_contagion_T, torch.exp)

        # A_{i,j} = relative frequency of interaction across regions
        #           -- for now, we're just using self.adjacency_matrix
        # self.connectivity = nn.Parameter(torch.eye(num_regions))

    def detection_p(self, t):
        return self.detection_rate[t] if t < self.num_days else self.detection_rate[-1]

    def recovery_p(self, t):
        return self.recovery_rate[t] if t < self.num_days else self.recovery_rate[-1]

    def contagion_I_p(self, t):
        return self.contagion_I[t] if t < self.num_days else self.contagion_I[-1]

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
        for compartment, out_flows in FLOWS_GRAPH.items():
            ns = state[compartment]
            probs = self.probs_from(compartment, state, t)
            values = torch.stack([getattr(flows, name) for name in out_flows], 1)
            logp += self.log_prob_for_flow(ns, probs, values)
        return logp

    def log_prob_for_flow(
        self,
        ns: torch.Tensor,
        probs: torch.Tensor,
        flow_vals: torch.Tensor,
    ):
        raise NotImplementedError("subclass should implement log_prob_for_flow")

    def sample_one_step(self, state: State, t: int) -> State:
        """
        Runs the model from ``history``; this model is Markov and
        therefore only looks at the *last* day of ``history``.
        """
        with torch.no_grad():
            flow_dict = {}
            for compartment, out_flows in FLOWS_GRAPH.items():
                ns = state[compartment]
                probs = self.probs_from(compartment, state, t)
                values = self.sample_one_flow(ns, probs).squeeze(0).t()
                for name, val in zip(out_flows, values):
                    flow_dict[name] = val

            return state.add_flow(Flows(**flow_dict))

    def sample_one_flow(self, ns: torch.Tensor, probs: torch.Tensor):
        raise NotImplementedError("subclass should implement sample_one_flow")

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
        with torch.no_grad():
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
        with torch.no_grad():
            max_day = max(days)
            samps = np.empty(
                (len(days), self.num_regions, len(History.fields), num_samples)
            )
            # TODO: should we avoid wrapping into a History object and then unwrapping?
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

        infectious_pops = self.contagion_I_p(t) * (self.adjacency_matrix @ state.I)
        total_pops = self.adjacency_matrix @ state.N
        return -torch.expm1(-infectious_pops / total_pops).unsqueeze(1)

    def probs_from_E(self, state: State, t: int) -> torch.Tensor:
        # E to I
        return self.decay_E.unsqueeze(0).unsqueeze(1)

    def probs_from_I(self, state: State, t: int) -> torch.Tensor:
        # I to T,U
        prob_I_out = self.decay_I.unsqueeze(0).unsqueeze(1)  # [1, 1]
        detection_rate = self.detection_p(t)  # [num_regions]
        return prob_I_out * torch.stack([detection_rate, 1 - detection_rate], dim=1)

    def probs_from_T(self, state: State, t: int) -> torch.Tensor:
        # T to R,D
        prob_T_out = self.decay_T.unsqueeze(0).unsqueeze(1)
        recovery_rate = self.recovery_p(t)  # [num_regions]
        return prob_T_out * torch.stack([recovery_rate, 1 - recovery_rate], dim=1)

    _probs_from_fn = {
        "S": probs_from_S,
        "E": probs_from_E,
        "I": probs_from_I,
        "T": probs_from_T,
    }

    def probs_from(self, compartment: str, state: State, t: int):
        return self._probs_from_fn[compartment](self, state, t)


class SeiturdModelNormalsRejectionNaive(BaseSeiturdModel):
    """
    A SEITURD model using plain normal likelihoods moment-matched to the multinomial,
    and rejection sampling for sampling to avoid negative flows.
    Thus the probability density and sampling do *not* agree with one another.
    """

    _cls_lookup = {1: UnivariateNormal, 2: BivariateNormal, 3: TrivariateNormal}

    def get_dist(self, ns: torch.Tensor, probs: torch.Tensor, fudge: float = 1e-7):
        means, vars = multinomial_moments(ns, probs, fudge=fudge)
        return self._cls_lookup[probs.shape[-1]](means, vars)

    def log_prob_for_flow(
        self,
        ns: torch.Tensor,
        probs: torch.Tensor,
        flow_vals: torch.Tensor,
        fudge: float = 1e-7,
    ):
        return self.get_dist(ns, probs, fudge).log_prob(flow_vals)

    def sample_one_flow(
        self, ns: torch.Tensor, probs: torch.Tensor, fudge: float = 1e-7
    ):
        dist = self.get_dist(ns, probs, fudge)
        while True:
            bits = dist.sample()
            if (bits >= 0).all():
                return bits.squeeze(0)


class SeiturdModelDirichlet(BaseSeiturdModel):
    """
    A SEITURD model using scaled Dirichlet distributions
    moment-matched to the multinomials.

    Specifically, for a Multinomial(n, p) with p a vector of length k,
    we use  n D where D ~ Dirichlet(s p), with s described in a moment;
    this has mean as desired
      E[n D] = n * s p / sum(s p) = n p,
    and covariance
      Cov(n D) = n^2 (diag(p) - p p^T) / (sum(s p) + 1)
               = n^2 (diag(p) - p p^T) / (s + 1)
               = n (diag(p) - p p^T) * n/(s+1).
    When (as usual) n > 1,
    we pick s = n - 1 to get an exact moment match to the multinomial distribution.
    If n <= 1 -- recall HistoryWithSoftmax only enforces n > 0 -- this isn't possible.
    So we instead use s = max(n - 1, eps),
    so that for n <= 1 + eps the variances are all off by a factor of n/(1+eps).
    I'm not too worried about it; this is easiest, and shouldn't be too common.
    The support is in any case is x_i in (0, n) with sum(x_i) = n.

    XXX: CURRENTLY DOES NOT WORK.
         That is, it probably *would* work, except that when we initialize
         the flows, it actually starts out with negative flows,
         which aren't in the support of a Dirichlet. 😅
    """

    def get_D_dist(
        self,
        ns: torch.Tensor,
        probs: torch.Tensor,
        eps: float = 0.01,
        min_alpha: float = 1e-7,
    ):
        # pad things out with the "self-flow"
        probs = torch.cat((probs, 1 - probs.sum(-1, keepdims=True)), -1)

        s_vals = (ns - 1).clamp(min=eps)
        alphas = (s_vals.unsqueeze(1) * probs).clamp(min=min_alpha)
        return Dirichlet(alphas)

    def log_prob_for_flow(
        self,
        ns: torch.Tensor,
        probs: torch.Tensor,  # shape [n_regions, n_flows]
        flow_vals: torch.Tensor,
        **kwargs
    ):
        # pad things out with the "self-flow"
        self_flows = ns.unsqueeze(-1) - flow_vals.sum(-1, keepdims=True)
        flow_vals = torch.cat((flow_vals, self_flows), -1)

        D_dist = self.get_D_dist(ns, probs, **kwargs)
        # change of variables: density of n D is 1/n times relevant density of D
        flow_Ds = flow_vals / ns.unsqueeze(-1)
        return D_dist.log_prob(flow_Ds) - torch.log(ns)

    def sample_one_flow(self, ns: torch.Tensor, probs: torch.Tensor, **kwargs):
        Ds = self.get_D_dist(ns, probs, **kwargs).sample()
        return ns.unsqueeze(0).unsqueeze(-1) * Ds


SeiturdModel = SeiturdModelNormalsRejectionNaive  # default implementation


def multinomial_moments(
    n: torch.Tensor, p: torch.Tensor, fudge: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes mean and covariance of a multinomial.

    Arguments:
      - n, number of trials, of shape [n_regions]
      - p, probabilities, of shape [n_regions, n_outs].
        p should be nonnegative, and sum to at most 1.
      - fudge, scalar to add to covariance diagonals to ensure it's p.d.
    Returns:
      - mean of shape [n_regions, n_outs]
      - cov of shape [n_regions, n_outs, n_outs]
    """
    if (n < 0).detach().any():
        raise ValueError("negative n")

    mean = n.unsqueeze(1) * p
    p_outer = p.unsqueeze(2) * p.unsqueeze(1)
    cov = torch.diag_embed(mean) - n[:, np.newaxis, np.newaxis] * p_outer
    fud_m = fudge * torch.eye(cov.shape[1], out=torch.empty_like(cov))[np.newaxis, :, :]
    return mean, cov + fud_m
