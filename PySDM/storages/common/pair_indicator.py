from typing import Type, TypeVar

from PySDM.storages.common.backend import PairBackend
from PySDM.storages.common.storage import PairIndicator, Storage

BackendType = TypeVar("BackendType", bound=PairBackend)


def pair_indicator(backend: BackendType, storage_cls: Type[Storage]):
    """
    Creates a specialised storage indicating whether the particle is the first in the pair.

    Parameters
    ----------
    backend : BackendType
        backend to be used for the storage
    storage_cls : Type[Storage]
        storage class to be used for the storage

    Returns
    -------
    Type[PairIndicator]
        specialised storage indicating whether the particle is the first in the pair
    """

    class _PairIndicator(PairIndicator):
        """
        The internal storage of the indicator.
        """

        def __init__(self, length):
            """
            Parameters
            ----------
            length : int
                number of particles

            Returns
            -------
            """
            assert isinstance(
                length, int
            ), "The length of the indicator must be integer."
            self.indicator = storage_cls.empty(length, dtype=storage_cls.BOOL)
            self.length = length

        def __len__(self) -> int:
            """
            Returns
            -------
            length of the indicator
            """
            return self.length

        def update(self, cell_start, cell_idx, cell_id):
            """
            Updates the indicator.
            """
            backend.find_pairs(cell_start, self, cell_id, cell_idx, cell_id.idx)
            self.length = len(cell_id)

    return _PairIndicator
