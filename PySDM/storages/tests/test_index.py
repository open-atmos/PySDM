import numpy as np
import pytest


def test_init(storage_class, index_class):
    # Act
    data = storage_class.from_ndarray(np.array([1, 2, 3])).data
    idx = index_class(data, 3)

    # Assert
    assert len(idx) == idx.length == 3
    assert isinstance(idx.length, index_class.INT)
    np.testing.assert_allclose(idx.to_ndarray(), np.array([1, 2, 3]))


def test_identity_index(index_class):
    # Act
    idx = index_class.identity_index(4)

    # Assert
    np.testing.assert_allclose(idx.to_ndarray(), np.arange(4))
    assert idx.dtype == idx.INT
    assert idx.shape == (4,)


def test_reset_index(index_class):
    # Arrange
    random_array = np.arange(0, 9)
    np.random.shuffle(random_array)
    idx = index_class.from_ndarray(random_array)

    # Act
    idx.reset_index()

    # Test
    np.testing.assert_allclose(idx.to_ndarray(), np.arange(0, 9))


def test_empty(index_class):
    with pytest.raises(
        TypeError, match="'Index' class cannot be instantiated as empty."
    ):
        index_class.empty()


def test_from_ndarray(index_class):
    idx = index_class.from_ndarray(np.asarray([1, 2, 3]))
    assert idx.dtype == index_class.INT
    assert idx.length == 3
    assert isinstance(idx.length, idx.INT)


@pytest.mark.parametrize(
    "storage_class",
    [
        ("PySDM.storages.numba.storage", "Storage"),
    ],
    indirect=True,
)
def test_sort_by_key(storage_class, index_class):
    random_array = np.arange(0, 9)
    np.random.shuffle(random_array)
    idx = index_class.from_ndarray(np.random.randint(0, 9, 9))
    idx.sort_by_key(index_class.from_ndarray(random_array))
    np.testing.assert_allclose(random_array[idx.data], np.sort(random_array)[::-1])


def test_remove_zero_n_or_flagged(storage_class, index_class):
    from PySDM.storages.common.indexed import indexed

    # Arrange
    n_sd = 44
    idx = index_class.identity_index(n_sd)
    data = np.ones(n_sd).astype(np.int64)
    data[0], data[n_sd // 2], data[-1] = 0, 0, 0
    data = storage_class.from_ndarray(data)
    data = indexed(storage_class).indexed(storage=data, idx=idx)

    # Act
    idx.remove_zero_n_or_flagged(data)
    # Assert
    assert len(idx) == n_sd - 3
    assert (data.to_ndarray(raw=True)[idx.to_ndarray()[: len(idx)]] > 0).all()
