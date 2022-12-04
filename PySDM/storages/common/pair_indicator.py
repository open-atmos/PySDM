from typing import Type, TypeVar

from PySDM.storages.common.backend import PairBackend
from PySDM.storages.common.storage import PairIndicator, Storage

BackendType = TypeVar("BackendType", bound=PairBackend)


def pair_indicator(backend: BackendType, storage_cls: Type[Storage]):
    class _PairIndicator(PairIndicator):
        def __init__(self, length):
            assert isinstance(
                length, int
            ), "The length of the indicator must be integer."
            self.indicator = storage_cls.empty(length, dtype=storage_cls.BOOL)
            self.length = length

        def __len__(self):
            return self.length

        def update(self, cell_start, cell_idx, cell_id):
            backend.find_pairs(cell_start, self, cell_id, cell_idx, cell_id.idx)
            self.length = len(cell_id)

    return _PairIndicator
