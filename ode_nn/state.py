"""
The ``State`` object tracks populations of individuals in a compartment model.
"""
from typing import Dict, Iterable, Union

import torch


class SmartState:
    """
    The popuations of individuals in a compartment model. Under the hood,
    this class tracks a ``torch.Tensor`` of shape
    ``(n_days, n_populations, n_regions)``. This class enables an API
    in which we can use ints and slices to obtain the tensor components
    directly, and strings to obtain the tensor for a single population.:

    .. code-block:: python

      a_state: State = ...
      S = a_state["S"]  # shape is (n_days, n_regions)
      state_slice = a_state[3:7, :, 10:20]  # shape (6, n_populations, 10)

    Args:
      populations (str, optional): default is 'SEIR'.
        A string of unique characters for the populations.
      n_days (int, optional): default is 100
      n_regions (int, optional): default is 50
    """

    def __init__(
        self,
        populations: str = "SEIR",
        n_days: int = 100,
        n_regions: int = 50,
    ) -> None:
        assert n_days > 0
        assert n_regions > 0

        # Unpack and check the populations
        msg = f"populations must be unique {populations}"
        assert len(set(populations)) == len(populations), msg
        self.populations = populations
        self._pop_set = set(populations)

        # Record the mapping of the population to its position
        self.mapping: Dict[Union[int, str], Union[int, str]] = dict()
        for k, pop in enumerate(populations):
            self.mapping[k] = pop
            self.mapping[pop] = k

        # Save attriubutes
        self.n_populations = len(populations)
        self.n_days = n_days
        self.n_regions = n_regions

        # Tensor for the state
        self.tensor = torch.zeros((n_days, self.n_populations, n_regions))

    def __getitem__(self, key: Union[str, int, Iterable]) -> torch.Tensor:
        # Handle pop name
        if isinstance(key, str):
            assert key in self._pop_set, f"invalid population {key}"
            return self.tensor[:, self.mapping[key]]
        else:  # Let torch figure it out
            return self.tensor[key]

    @property
    def N(self) -> torch.Tensor:
        """
        The state summed over the population dimension (dim 2).

        Returns:
          the total populations summed over all states
        """
        return self.tensor.sum(axis=1)
