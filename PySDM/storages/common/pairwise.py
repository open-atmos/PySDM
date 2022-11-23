"""
specialised storage equipped with particle pair-handling methods
"""
from typing import Type, TypeVar

from PySDM.storages.common.backend import PairBackend
from PySDM.storages.common.storage import Pairwise, Storage

BackendType = TypeVar("BackendType", bound=PairBackend)


def pairwise(backend: BackendType, storage_cls: Type[Storage]):
    assert issubclass(storage_cls, Storage)

    class _PairwiseStorage(storage_cls, Pairwise):
        def distance(self, other, is_first_in_pair):
            backend.distance_pair(self, other, is_first_in_pair, other.idx)

        def max(self, other, is_first_in_pair):
            backend.max_pair(self, other, is_first_in_pair, other.idx)

        def min(self, other, is_first_in_pair):
            backend.min_pair(self, other, is_first_in_pair, other.idx)

        def sort(self, other, is_first_in_pair):
            backend.sort_pair(self, other, is_first_in_pair, other.idx)

        def sum(self, other, is_first_in_pair):
            backend.sum_pair(self, other, is_first_in_pair, other.idx)

        def multiply(self, other, is_first_in_pair):
            backend.multiply_pair(self, other, is_first_in_pair, other.idx)

    return _PairwiseStorage
