from contextlib import nullcontext

import numpy as np
import pytest

from PySDM.storages.common.storage import StorageSignature
from PySDM.storages.numba.random import Random
from PySDM.storages.numba.storage import Storage


def test_storage_init():
    data = np.ones(5, dtype=Storage.INT)
    storage = Storage(StorageSignature(data, data.shape, data.dtype))
    assert storage.shape == (5,)
    assert storage.dtype == storage.INT
    np.testing.assert_allclose(storage.data, data)
    assert storage.signature == StorageSignature(data, data.shape, data.dtype)
    assert len(storage) == 5


@pytest.mark.parametrize(
    ["shape_and_dtype", "expected_data"],
    [
        ((5, Storage.INT), np.full(5, -1, dtype=Storage.INT)),
        ((3, Storage.FLOAT), np.full(3, -1.0, dtype=Storage.FLOAT)),
        ((11, Storage.BOOL), np.full(11, -1, dtype=Storage.BOOL)),
    ],
)
def test_storage_empty(shape_and_dtype, expected_data):
    storage = Storage.empty(*shape_and_dtype)
    assert storage.dtype == expected_data.dtype
    assert storage.shape == expected_data.shape
    np.testing.assert_allclose(storage.data, expected_data)


@pytest.mark.parametrize(
    ["array", "expected_data"],
    [
        (
            np.full(5, -1, dtype=np.int32),
            np.full(5, -1, dtype=Storage.INT),
        ),
        (np.zeros(13, np.float16), np.zeros(13, dtype=Storage.FLOAT)),
        (
            np.ones(11, np.bool8),
            np.ones(11, Storage.BOOL),
        ),
    ],
)
def test_storage_from_array(array, expected_data):
    storage = Storage.from_ndarray(array)
    assert storage.dtype == expected_data.dtype
    assert storage.shape == expected_data.shape
    np.testing.assert_allclose(storage.data, expected_data)


def test_storage_upload():
    data = np.random.rand(5)
    storage = Storage.from_ndarray(np.random.rand(5))
    storage.upload(data)
    np.testing.assert_allclose(storage.data, data)


def test_storage_urand():
    random_gen = Random(5, 1)
    zeros = np.zeros((4, 5), dtype=Storage.FLOAT)
    storage = Storage(StorageSignature(zeros.copy(), zeros.shape, zeros.dtype))
    storage.urand(random_gen)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, storage.data, zeros
    )


def test_storage_to_ndarray():
    data = np.random.rand(5, 5)
    storage = Storage(StorageSignature(data, data.shape, Storage.FLOAT))
    np.testing.assert_allclose(storage.to_ndarray(), data)
    assert data is not storage.to_ndarray()


@pytest.mark.parametrize(
    ["array_size", "expected_context", "expected_value"],
    [
        ((1,), nullcontext(), True),
        (
            (2, 3),
            pytest.raises(
                NotImplementedError, match="Logic value of array is ambiguous."
            ),
            False,
        ),
    ],
)
def test_bool(array_size, expected_context, expected_value):
    storage = Storage.from_ndarray(np.random.random(array_size))
    with expected_context:
        assert expected_value == bool(storage)


def test_detach():
    data = np.arange(1, 9, dtype=Storage.INT)
    assert data.base is None
    sliced_data = data[3:8]
    assert sliced_data.base is not None
    storage = Storage(
        StorageSignature(
            data=sliced_data, shape=sliced_data.shape, dtype=sliced_data.dtype
        )
    )
    assert storage.data.base is not None
    storage.detach()
    assert storage.data.base is None


def test_amin():
    data = np.arange(1, 11).reshape(5, 2)
    storage = Storage.from_ndarray(data)
    assert storage.amin() == 1


def test_all():
    assert Storage.from_ndarray(np.full((3, 4), fill_value=True, dtype=np.bool_)).all()
    assert not Storage.from_ndarray(
        np.full((3, 5), fill_value=False, dtype=np.bool_)
    ).all()


def test_floor():
    storage = Storage.from_ndarray(np.asarray([-0.5, 1.25, 2.1]))
    storage = storage.floor()
    np.testing.assert_allclose(storage.data, [-1, 1, 2])


def test_floor_out_of_place():
    storage = Storage.from_ndarray(np.asarray([-0.5, 1.25, 2.1]))
    storage = storage.floor(Storage.from_ndarray(np.asarray([-1.9, 0.2, 1.3])))
    np.testing.assert_allclose(storage.data, [-2, 0, 1])


def test_product():
    data = np.asarray([1, 2, 3])
    storage = Storage.from_ndarray(data)
    storage = storage.product(storage, storage)
    np.testing.assert_allclose(storage.data, data**2)

    storage = storage.product(storage, data)
    np.testing.assert_allclose(storage.data, data**3)


def test_ratio():
    divident = np.asarray([2.4, 5.6, 3.2])
    divisor = np.asarray([9.8, 2.4, 12.3])
    storage = Storage.from_ndarray(np.ones_like(divident))
    storage.ratio(Storage.from_ndarray(divident), Storage.from_ndarray(divisor))
    np.testing.assert_allclose(storage.data, divident / divisor)


def test_divide_if_not_zero():
    divident = np.asarray([2.4, 3.4, 3.2])
    divisor = np.asarray([9.8, 1, 12.3])
    storage = Storage.from_ndarray(divident)
    divisor_with_0 = divisor.copy()
    divisor_with_0[1] = 0
    storage.divide_if_not_zero(Storage.from_ndarray(divisor_with_0))
    np.testing.assert_allclose(storage.data, divident / divisor)


def test_sum():
    a = Storage.from_ndarray(np.ones(3))
    b = Storage.from_ndarray(np.ones(3))
    b *= 2
    storage = Storage.empty(3, a.dtype)
    storage.sum(a, b)
    np.testing.assert_allclose(storage.data, np.ones(3) * 3)


def test_ravel():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    storage = Storage.empty(6, data.dtype)
    storage.ravel(data)
    np.testing.assert_allclose(storage.data, np.ravel(data))
