import numpy as np

from PySDM.storages.common.pair_indicator import pair_indicator
from PySDM.storages.numba.backend.pair import PairBackend
from PySDM.storages.numba.storage import Storage


def test_init():
    # Arrange
    backend = PairBackend()
    pair_indicator_cls = pair_indicator(backend, storage_cls=Storage)
    pair_indicator_instance = pair_indicator_cls(6)

    # Assert
    assert len(pair_indicator_instance) == 6
    assert np.all(pair_indicator_instance.indicator.data)
