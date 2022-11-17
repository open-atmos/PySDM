# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.storages.common.index import index
from PySDM.storages.common.indexed import indexed
from PySDM.storages.numba.backend.index import IndexBackend
from PySDM.storages.numba.storage import Storage


def test_init():
    # Arrange
    backend = IndexBackend()
    index_cls = index(backend, Storage)

    # Act
    idx = index_cls(np.array([1, 2, 3]), 3)

    # Assert
    assert len(idx) == idx.length == 3
    assert isinstance(idx.length, index_cls.INT)
    np.testing.assert_allclose(idx.data, np.array([1, 2, 3]))


def test_identity_index():
    # Arrange
    backend = IndexBackend()
    index_cls = index(backend, Storage)

    # Act
    idx = index_cls.identity_index(4)

    # Assert
    np.testing.assert_allclose(idx.data, np.arange(4))
    assert idx.dtype == idx.INT
    assert idx.shape == (4,)


def test_reset_index():
    # Arrange
    backend = IndexBackend()
    index_cls = index(backend, Storage)
    random_array = np.random.randint(0, 9, 9)
    idx = index_cls.from_ndarray(random_array)

    # Act
    idx.reset_index()

    # Test
    np.testing.assert_allclose(idx.data, np.arange(0, 9))


def test_empty():
    backend = IndexBackend()
    index_cls = index(backend, Storage)
    with pytest.raises(
        TypeError, match="'Index' class cannot be instantiated as empty."
    ):
        index_cls.empty()


def test_from_ndarray():
    backend = IndexBackend()
    index_cls = index(backend, Storage)
    idx = index_cls.from_ndarray(np.asarray([1, 2, 3]))
    assert idx.dtype == index_cls.INT
    assert idx.length == 3
    assert isinstance(idx.length, idx.INT)


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
