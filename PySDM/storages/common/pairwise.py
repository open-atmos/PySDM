"""
specialised storage equipped with particle pair-handling methods
"""
from typing import Type, TypeVar

from PySDM.storages.common.backend import PairBackend
from PySDM.storages.common.storage import Indexed, PairIndicator, Pairwise, Storage

BackendType = TypeVar("BackendType", bound=PairBackend)


def pairwise(backend: BackendType, storage_cls: Type[Storage]):
    """
    Creates a specialised storage equipped with particle pair-handling methods.

    Parameters
    ----------
    backend : BackendType
        backend to be used for the storage
    storage_cls : Type[Storage]
        storage class to be used for the storage

    Returns
    -------
    Type[Pairwise]
        specialised storage equipped with particle pair-handling methods
    """
    assert issubclass(storage_cls, Storage)

    class _PairwiseStorage(storage_cls, Pairwise):
        def distance(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
            """
            Calculates the distance between particles in the pair.

            Parameters
            ----------
            other : Indexed
                storage with the second particle in the pair
            is_first_in_pair : PairIndicator
                storage indicating whether the particle is the first in the pair

            Returns
            -------
            None
            """
            backend.distance_pair(self, other, is_first_in_pair, other.idx)

        def max(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
            """
            Calculates the maximum of the particle and the second particle in the pair.

            Parameters
            ----------
            other : Indexed
                storage with the second particle in the pair
            is_first_in_pair : PairIndicator
                storage indicating whether the particle is the first in the pair

            Returns
            -------
            None
            """
            backend.max_pair(self, other, is_first_in_pair, other.idx)

        def min(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
            """
            Calculates the minimum of the particle and the second particle in the pair.

            Parameters
            ----------
            other : Indexed
                storage with the second particle in the pair
            is_first_in_pair : PairIndicator
                storage indicating whether the particle is the first in the pair

            Returns
            -------
            None
            """
            backend.min_pair(self, other, is_first_in_pair, other.idx)

        def sort(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
            """
            Sorts the particles in the pair.

            Parameters
            ----------
            other : Indexed
                storage with the second particle in the pair
            is_first_in_pair : PairIndicator
                storage indicating whether the particle is the first in the pair

            Returns
            -------
            None
            """
            backend.sort_pair(self, other, is_first_in_pair, other.idx)

        def sum(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
            """
            Calculates the sum of the particle and the second particle in the pair.

            Parameters
            ----------
            other : Indexed
                storage with the second particle in the pair
            is_first_in_pair : PairIndicator
                storage indicating whether the particle is the first in the pair

            Returns
            -------
            None
            """
            backend.sum_pair(self, other, is_first_in_pair, other.idx)

        def multiply(self, other: Indexed, is_first_in_pair: PairIndicator) -> None:
            """
            Calculates the product of the particle and the second particle in the pair.

            Parameters
            ----------
            other : Indexed
                storage with the second particle in the pair
            is_first_in_pair : PairIndicator
                storage indicating whether the particle is the first in the pair

            Returns
            -------
            None
            """
            backend.multiply_pair(self, other, is_first_in_pair, other.idx)

    return _PairwiseStorage
