import numpy as np

from PySDM.storages.common.pair_indicator import pair_indicator
from PySDM.storages.common.pairwise import pairwise
from PySDM.storages.numba.backend.pair import PairBackend
from PySDM.storages.numba.storage import Storage


def test_init():
    # Arrange
    backend = PairBackend()
    pair_indicator_cls = pair_indicator(backend, storage_cls=Storage)
    pair_indicator_instance = pair_indicator_cls(6)
    pairwise_storage_cls = pairwise(backend, storage_cls=Storage)
