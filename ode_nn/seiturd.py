import random
from collections import namedtuple
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

State = namedtuple("State", list("SEITURD"))


class Seiturd(nn.Module):
    """
    dataset_shape:
    mask_adjacency:

    .. note::
      To send a `Seiturd` to a device, instantiate it and then do `model.to(device)`.
    """

    def __init__(
        self,
        dataset_shape: List[int],
        mask_adjacency: bool = True,
    ):
        super().__init__()

        num_days, _, num_regions = dataset_shape
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
        self.detection_rate = nn.Parameter(torch.rand((num_days, num_regions)) / 10)
        # r_{i,t} = recovery rate
        self.recovery_rate = nn.Parameter(torch.rand((num_days, num_regions)) / 10)
        # beta_{i,t} = probability of infection per interaction with I person
        self.contagion_I = nn.Parameter(torch.rand((num_days, num_regions)) / 10)
        # eps_{i,t} = probability of infection per interaction with T person
        self.contagion_T = nn.Parameter(torch.rand((num_days, num_regions)) / 10)
        # A_{i,j} = percentage of infected people in state j who interact with
        # each susceptible person of state i
        self.connectivity = nn.Parameter(torch.eye(num_regions))

        # S = susceptible population
        self.latent_S = nn.Parameter(torch.tensor([0.5] * num_regions))
        # E = exposed population in the non-contagious incubation period
        self.latent_E = nn.Parameter(torch.tensor([0.5] * num_regions))
        # I = infected population (contagious but not tested)
        self.latent_I = nn.Parameter(torch.tensor([0.5] * num_regions))
        # T = infected population that has tested positive
        self.latent_T = nn.Parameter(torch.tensor([0.5] * num_regions))
        # U = undetected population that has either self-quarantined or recovered
        self.latent_U = nn.Parameter(torch.tensor([0.5] * num_regions))
        # R = recovered population that has tested positive
        self.latent_R = nn.Parameter(torch.tensor([0.5] * num_regions))
        # D = death toll
        self.latent_D = nn.Parameter(torch.tensor([0.5] * num_regions))

    @property
    def decay_E(self) -> torch.Tensor:
        return torch.exp(self.ln_decay_E)

    @property
    def decay_I(self) -> torch.Tensor:
        return torch.exp(self.ln_decay_I)

    @property
    def decay_T(self) -> torch.Tensor:
        return torch.exp(self.ln_decay_T)

    def one_step(self, t_initial: int, state_initial: State) -> State:
        dS = self.change_S(state_initial, t_initial)
        dE = self.change_E(state_initial, t_initial)
        dI = self.change_I(state_initial)
        dT = self.change_T(state_initial)
        dU = self.change_U(state_initial)
        dR = self.change_R(state_initial)
        dD = dR * self.fraction_D(t_initial)

        new_state = State(
            state_initial.S + dS,
            state_initial.E + dE,
            state_initial.I + dI,
            state_initial.T + dT,
            state_initial.U + dU,
            state_initial.R + dR,
            state_initial.D + dD,
        )
        return new_state

    def many_steps(
        self, t_initial: int, t_final: int, state_initial: State
    ) -> List[State]:
        states = np.empty(t_final - t_initial, dtype=State)
        cur_state = state_initial  # is this operation free?
        for t in range(t_initial, t_final):
            cur_state = self.one_step(t_initial, cur_state)
            states[t - t_initial] = cur_state
        return states

    def forward(self, num_steps):
        return 0

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
    # TODO: the mask needs to be defined
    def prob_S_E(self, I_t, t):
        return self.contagion_I[t, :] * (
            torch.mm(self.A, (I_t).reshape(-1, 1)).squeeze(1)
        )

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

    def change_I(self, state):
        return self.prob_E_I() * state.E - self.prob_I_out() * state.I

    def change_T(self, state):
        return self.prob_I_T() * state.I - self.prob_T_out() * state.T

    def change_U(self, state):
        return self.prob_I_U() * state.I

    def change_R(self, state):
        return self.prob_T_R() * state.T

    def change_D(self, state):
        return self.prob_T_D() * state.T
