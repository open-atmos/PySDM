import numpy as np

from PySDM.storages.common.index import index
from PySDM.storages.common.indexed import indexed
from PySDM.storages.common.storage import StorageSignature
from PySDM.storages.numba.backend.index import IndexBackend
from PySDM.storages.numba.storage import Storage


def test_init():
    # Arrange
    backend = IndexBackend()
    index_cls = index(backend, Storage)
    indexed_cls = indexed(Storage)
    idx_data = np.arange(3)
    idx_storage = index_cls.from_ndarray(idx_data)
    np.random.shuffle(idx_data)
    data = np.array([4, 3, 6])

    # Act
    indexed_storage = indexed_cls(
        idx_storage, StorageSignature(data, data.shape, data.dtype)
    )

    # Assert
    assert indexed_storage.idx is not None
    assert len(indexed_storage) == len(idx_storage) == 3
    np.testing.assert_allclose(indexed_storage.data, np.array([4, 3, 6]))
