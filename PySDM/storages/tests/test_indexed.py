import numpy as np

from PySDM.storages.common.storage import StorageSignature


def test_init(storage_class, index_class, indexed_class):
    # Arrange
    idx_data = np.arange(3)
    idx_storage = index_class.from_ndarray(idx_data)
    np.random.shuffle(idx_data)
    data = storage_class.from_ndarray(np.array([4, 3, 6])).data

    # Act
    indexed_storage = indexed_class(
        idx_storage, StorageSignature(data, 3, storage_class.INT)
    )

    # Assert
    assert indexed_storage.idx is not None
    assert len(indexed_storage) == len(idx_storage) == 3
    np.testing.assert_allclose(indexed_storage.to_ndarray(), np.array([4, 3, 6]))


def test_getitem(storage_class, index_class, indexed_class):
    # Arrange
    idx_data = np.arange(6)
    idx_storage = index_class.from_ndarray(idx_data)
    np.random.shuffle(idx_data)
    storage = storage_class.from_ndarray(np.random.randint(0, 9, 6))

    # Act
    indexed_storage = indexed_class(
        idx_storage, StorageSignature(storage.data, 6, storage_class.INT)
    )

    # Assert
    assert indexed_storage.idx is not None
    assert indexed_storage[0] == storage.data[0]

    sliced = indexed_storage[3:5]
    assert isinstance(sliced, indexed_class)
    assert sliced.idx is not None
    sliced_idx_data = sliced.idx.to_ndarray()
    idx = sliced_idx_data[(sliced_idx_data >= 3) & (sliced_idx_data < 5)]
    np.testing.assert_allclose(storage.to_ndarray()[idx], storage.to_ndarray()[3:5])


def test_indexed_and_empty(storage_class, index_class, indexed_class):
    from PySDM.storages.numba.storage import Storage as NumbaStorage

    # Arrange
    idx = index_class.from_ndarray(np.arange(3))
    indexed_and_empty = indexed_class.indexed_and_empty(idx, 3, indexed_class.INT)

    # Act
    if storage_class == NumbaStorage:
        np.testing.assert_allclose(
            indexed_and_empty.to_ndarray(), np.full(3, -1, dtype=storage_class.INT)
        )
    np.testing.assert_allclose(indexed_and_empty.idx.to_ndarray(), idx.to_ndarray())


def test_indexed_from_ndarray(storage_class, index_class, indexed_class):
    # Arrange
    idx = index_class.from_ndarray(np.arange(3))
    data = np.random.randint(-1, 8, 3)
    indexed_from_ndarray = indexed_class.indexed_from_ndarray(idx, data)

    np.testing.assert_allclose(indexed_from_ndarray.to_ndarray(), data)
    np.testing.assert_allclose(indexed_from_ndarray.idx.to_ndarray(), idx.to_ndarray())


def test_to_ndarray(storage_class, index_class, indexed_class):
    # Arrange
    idx_data = np.arange(6)
    np.random.shuffle(idx_data)
    idx = index_class.from_ndarray(idx_data)

    data = np.random.randint(-12, 16, 6)
    indexed_from_ndarray = indexed_class.indexed_from_ndarray(idx, data)
    data_ = indexed_from_ndarray.to_ndarray(raw=True)
    np.testing.assert_allclose(data_, data)

    data_ = indexed_from_ndarray.to_ndarray(raw=False)
    np.testing.assert_allclose(data_.data, data[idx_data])
