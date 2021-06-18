"""
The ``State`` object tracks populations of individuals in a compartment model.
"""
from typing import Dict, Iterable, Optional, Union

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
      requires_grad (bool, optional): default ``False``. flag to allow for
        gradients
      device (Optional[torch.device], optional): default ``None``. Send the
        :attr:`tensor` to the device
    """

    def __init__(
        self,
        populations: str = "SEIR",
        n_days: int = 100,
        n_regions: int = 50,
        requires_grad: bool = False,
        device: Optional[torch.device] = None,
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
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor = torch.zeros(
            (n_days, self.n_populations, n_regions),
            device=device,
            requires_grad=requires_grad,
        )

    def __getitem__(self, key: Union[str, int, Iterable]) -> torch.Tensor:
        """
        Index the underlying tensor intelligently. If a string was passed in,
        then return the ``(n_days, n_regions)`` tensor associated with that
        population. If any other kind of index was passed int, index the
        :attr:`tensor` attribute directly.

        Args:
          key (Union[str, int, Iterable]): either a population name or
            numerical values to index the tensor

        Returns:
          the population(s) corresponding to the key
        """
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

    def __len__(self) -> int:
        return len(self.tensor)

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape
