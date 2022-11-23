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


def test_getitem():
    # Arrange
    backend = IndexBackend()
    index_cls = index(backend, Storage)
    indexed_cls = indexed(Storage)
    idx_data = np.arange(6)
    idx_storage = index_cls.from_ndarray(idx_data)
    np.random.shuffle(idx_data)
    data = np.random.randint(0, 9, 6)

    # Act
    indexed_storage = indexed_cls(
        idx_storage, StorageSignature(data, data.shape, data.dtype)
    )

    # Assert
    assert indexed_storage.idx is not None
    assert indexed_storage[0] == data[0]

    sliced = indexed_storage[3:5]
    assert isinstance(sliced, indexed_cls)
    assert sliced.idx is not None
    idx = sliced.idx.data[(sliced.idx.data >= 3) & (sliced.idx.data < 5)]
    np.testing.assert_allclose(data[idx], sliced.data)


def test_indexed_and_empty():
    backend = IndexBackend()
    index_cls = index(backend, Storage)
    indexed_cls = indexed(Storage)

    idx = index_cls.from_ndarray(np.arange(3))
    indexed_and_empty = indexed_cls.indexed_and_empty(idx, 3, indexed_cls.INT)

    np.testing.assert_allclose(
        indexed_and_empty.data, np.full(3, -1, dtype=Storage.INT)
    )
    np.testing.assert_allclose(indexed_and_empty.idx.data, idx.data)


def test_indexed_from_ndarray():
    backend = IndexBackend()
    index_cls = index(backend, Storage)
    indexed_cls = indexed(Storage)

    idx = index_cls.from_ndarray(np.arange(3))
    data = np.random.randint(-1, 8, 3)
    indexed_from_ndarray = indexed_cls.indexed_from_ndarray(idx, data)

    np.testing.assert_allclose(indexed_from_ndarray.data, data)
    np.testing.assert_allclose(indexed_from_ndarray.idx.data, idx.data)


def test_to_ndarray():
    backend = IndexBackend()
    index_cls = index(backend, Storage)
    indexed_cls = indexed(Storage)

    idx_data = np.arange(6)
    np.random.shuffle(idx_data)
    idx = index_cls.from_ndarray(idx_data)

    data = np.random.randint(-12, 16, 6)
    indexed_from_ndarray = indexed_cls.indexed_from_ndarray(idx, data)
    data_ = indexed_from_ndarray.to_ndarray(raw=True)
    np.testing.assert_allclose(data_, data)

    data_ = indexed_from_ndarray.to_ndarray(raw=False)
    np.testing.assert_allclose(data_.data, data[idx_data])
