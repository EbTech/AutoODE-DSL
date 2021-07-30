"""
Classes for handling populations.
"""

from __future__ import annotations

import datetime
from typing import NamedTuple, Optional

import numpy as np
import torch
from torch.nn import functional as F

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

    # As the transition model is a tree, flows are uniquely determined by delta,
    # since both have one fewer degree of freedom than the number of states.
    # To infer flows, proceed by eliminating one leaf at a time.
    def flows_to(self, other: State) -> Flows:
        delta = State(*[b - a for (a, b) in zip(self, other)])

        T_D = delta.D
        T_R = delta.R
        I_U = delta.U
        I_T = delta.T + T_D + T_R  # other.T == self.T - T_D - T_R + I_T
        E_I = delta.I + I_T + I_U  # other.I == self.I - I_T - I_U + E_I
        S_E = delta.E + E_I  # other.E == self.E - E_I + S_E

        return Flows(S_E=S_E, E_I=E_I, I_T=I_T, I_U=I_U, T_R=T_R, T_D=T_D)

    def add_flow(self, flows: Flows) -> State:
        S = self.S - flows.S_E
        E = self.E + flows.S_E - flows.E_I
        I = self.I + flows.E_I - flows.I_T - flows.I_U  # noqa: E741
        T = self.T + flows.I_T - flows.T_R - flows.T_D
        U = self.U + flows.I_U
        R = self.R + flows.T_R
        D = self.D + flows.T_D

        return State(S=S, E=E, I=I, T=T, U=U, R=R, D=D)


class Flows(NamedTuple):
    S_E: torch.Tensor
    E_I: torch.Tensor
    I_T: torch.Tensor
    I_U: torch.Tensor
    T_R: torch.Tensor
    T_D: torch.Tensor


class BaseHistory(torch.nn.Module):
    """
    Represents the values of the SEITURD subpopulations over a span of ``num_days``.

    This is an abstract base class, use a subclass to actually store the data
    somehow.
    """

    fields = "SEITURD"  # could avoid hardcoding these, not bothering for now
    # child class needs to implement S, E, I, etc properties

    def __init__(
        self,
        N: torch.Tensor,  # total pop, shape [num_regions]
        num_pos_and_alive: torch.Tensor,  # T + R, shape [num_days, num_regions]
        num_dead: torch.Tensor,  # D, shape [num_days, num_regions]
        # num_days: int,
        requires_grad: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.get_default_dtype()

        # these are used in subclass __init__ but shouldn't really be "public"
        self._device = device
        self._requires_grad = requires_grad

        # copy the input data so we don't accidentally modify it in __setitem__
        def cast(t):
            return torch.tensor(t, device=device, dtype=dtype)

        self.N = cast(N)
        self.num_pos_and_alive = cast(num_pos_and_alive)  # total of T + R
        (self.num_days, self.num_regions) = self.num_pos_and_alive.shape
        assert self.N.shape == (self.num_regions,)
        self.num_dead = cast(num_dead)  # people in D
        assert num_dead.shape == (self.num_days, self.num_regions)

        # free_pop = S + E + I + U = N - (T + R + D)
        self.free_pop = self.N[np.newaxis, :] - self.num_pos_and_alive - self.num_dead

        # child class __init__ should use this to initialize the actual data now

    @classmethod
    def from_dataset(
        cls,
        dataset: C19Dataset,
        first_date: Optional[datetime.date] = None,
        last_date: Optional[datetime.date] = None,
        **kwargs,
    ):
        # must be a nicer way to do this....
        if first_date is None:
            first_date = datetime.date(1970, 1, 1)
        if last_date is None:
            last_date = datetime.date(3000, 1, 1)
        dates = dataset.df.reset_index().date
        which = ((dates >= first_date) & (dates <= last_date)).values

        TRD = dataset.tensor[which, 0, :]
        D = dataset.tensor[which, -1, :]
        TR = TRD - D
        return cls(N=dataset.pop_2018, num_pos_and_alive=TR, num_dead=D, **kwargs)

    def __getitem__(self, i: int):
        return State(**{n: getattr(self, n)[i] for n in self.fields})

    def __len__(self):
        return self.num_days

    def __repr__(self):
        return (
            f"<{type(self).__qualname__} object "
            f"({self.num_days} days, {self.num_regions} regions)>"
        )


class HistoryWithImplicits(BaseHistory):
    """
    The S state is implicit (S = N - E - I - T - U - R - D); this choice *does*
    affect the gradient descent direction slightly, but seems unlikely
    that would really matter.

    The R state is also implicit (R = num_pos_and_alive - T).

    This approach has no "built-in" protection against negative values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 7 SEITURD states minus 1 population constraint minus 2 data-enforced
        # constraints equals 4 states to fit
        self.data = torch.full(
            (4, self.num_days, self.num_regions),
            0.0,
            device=self._device,
            requires_grad=self._requires_grad,
        )

        # initializing to 0 makes the likelihoods also act weird
        # arbitrarily choose to divide available pops evenly among SEIU and TR
        # (remembering that S and R are implicit)
        self.data[[0, 1, 3], :, :] = self.free_pop[np.newaxis, :, :] / 4
        self.data[2, :, :] = self.num_pos_and_alive / 2

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

    def __setitem__(self, i: int, state: State):
        assert torch.allclose(state.N, self.N)
        self.num_pos_and_alive[i] = state.T + state.R
        self.num_dead[i] = state.D
        self.free_pop[i] = self.N - self.num_pos_and_alive[i] - self.num_dead[i]
        for name, state_val in zip(State._fields, state):
            if name in "SRD":  # implicit stuff
                continue
            # this is gross, but okay b/c of how EITU are implemented:
            # it's essentially, e.g., self.data[2][i] = state_val
            getattr(self, name)[i] = state_val

    # thoughts about gradients in this model:
    #
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


class HistoryWithSoftmax(BaseHistory):
    """
    We know S+E+I+U and T+R at all times; stores the splits between them as
    "logits" so that the sum is correct.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def make(n):
            # TODO - this doesn't make parameters
            return torch.nn.Parameter(
                torch.full(
                    (self.num_days, self.num_regions, n),
                    0.0,
                    device=self._device,
                    requires_grad=self._requires_grad,
                )
            )

        self.logits_SEIU = make(4)
        self.logits_TR = make(2)

    # this feels incredibly wasteful :/ - maybe implement caching...
    SEIU = property(
        lambda self: self.free_pop.unsqueeze(2) * F.softmax(self.logits_SEIU, dim=2)
    )
    S = property(lambda self: self.SEIU[..., 0])
    E = property(lambda self: self.SEIU[..., 1])
    I = property(lambda self: self.SEIU[..., 2])  # noqa: E741
    U = property(lambda self: self.SEIU[..., 3])

    TR = property(
        lambda self: self.num_pos_and_alive.unsqueeze(2)
        * F.softmax(self.logits_TR, dim=2)
    )
    T = property(lambda self: self.TR[..., 0])
    R = property(lambda self: self.TR[..., 1])

    D = property(lambda self: self.num_dead)

    def __setitem__(self, i: int, state: State):
        assert torch.allclose(state.N, self.N)

        self.num_pos_and_alive[i] = state.T + state.R
        self.num_dead[i] = state.D
        self.free_pop[i] = self.N - self.num_pos_and_alive[i] - self.num_dead[i]

        self.logits_SEIU[i] = torch.log(
            torch.stack([state.S, state.E, state.I, state.U], 1)
        )
        self.logits_TR[i] = torch.log(torch.stack([state.T, state.R], 1))


History = HistoryWithSoftmax  # default implementation
