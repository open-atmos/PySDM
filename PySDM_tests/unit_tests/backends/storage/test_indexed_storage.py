"""
Created at 05.10.2020
"""

import numpy as np
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


class TestIndexedStorage:

    @staticmethod
    def test_remove_zeros(backend):
        # Arrange
        n_sd = 44
        idx = backend.IndexedStorage.from_ndarray(np.arange(0, n_sd).astype(np.int64))
        data = np.ones(n_sd).astype(np.int64)
        data[0], data[n_sd // 2], data[-1] = 0, 0, 0
        data = backend.Storage.from_ndarray(data)
        data = backend.IndexedStorage.indexed(storage=data, idx=idx)

        # Act
        new_n_sd = backend.remove_zeros(data.data, idx.data, n_sd)

        # Assert
        assert new_n_sd == n_sd - 3
        assert (data.to_ndarray()[idx.to_ndarray()[:new_n_sd]] > 0).all()
