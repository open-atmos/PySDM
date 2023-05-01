# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM.backends.impl_common.index import make_Index
from PySDM.backends.impl_common.indexed_storage import make_IndexedStorage


class TestIndex:  # pylint: disable=too-few-public-methods
    @staticmethod
    def test_remove_zero_n_or_flagged(backend_class):
        # Arrange
        backend = backend_class()
        n_sd = 44
        idx = make_Index(backend).identity_index(n_sd)
        data = np.ones(n_sd).astype(np.int64)
        data[0], data[n_sd // 2], data[-1] = 0, 0, 0
        data = backend.Storage.from_ndarray(data)
        data = make_IndexedStorage(backend).indexed(storage=data, idx=idx)

        # Act
        idx.remove_zero_n_or_flagged(data)

        # Assert
        assert len(idx) == n_sd - 3
        assert (
            backend.Storage.to_ndarray(data)[idx.to_ndarray()[: len(idx)]] > 0
        ).all()
