# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM.storages.common.index import index
from PySDM.storages.common.indexed import indexed
from PySDM.storages.numba.backend.index import IndexBackend
from PySDM.storages.numba.storage import Storage


def test_remove_zero_n_or_flagged():
    # Arrange
    backend = IndexBackend()
    n_sd = 44
    idx = index(backend, Storage).identity_index(n_sd)
    data = np.ones(n_sd).astype(np.int64)
    data[0], data[n_sd // 2], data[-1] = 0, 0, 0
    data = Storage.from_ndarray(data)
    data = indexed(Storage).indexed(storage=data, idx=idx)

    # Act
    idx.remove_zero_n_or_flagged(data)
    # Assert
    assert len(idx) == n_sd - 3
    assert (data.to_ndarray(raw=True)[idx.to_ndarray()[: len(idx)]] > 0).all()
