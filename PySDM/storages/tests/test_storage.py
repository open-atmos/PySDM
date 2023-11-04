from contextlib import nullcontext

import numpy as np
import pytest

from PySDM.storages.common.storage import StorageSignature
from PySDM.storages.thrust_rtc.test_helpers.flag import fakeThrustRTC


@pytest.mark.parametrize(
    ["dtype", "size"],
    [
        ("INT", 5),
        ("FLOAT", 1),
        ("BOOL", 10),
    ],
)
def test_storage_init(storage_class, dtype, size):
    dtype = getattr(storage_class, dtype)
    data = np.ones(size, dtype=dtype)
    storage = storage_class.from_ndarray(data)
    assert storage.shape == (size,)
    assert storage.dtype == dtype
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, data)
    assert storage.signature.shape == data.shape
    assert storage.signature.dtype == data.dtype
    assert len(storage) == size


@pytest.mark.parametrize(
    ["shape", "dtype"],
    [
        (5, "INT"),
        (3, "FLOAT"),
        (11, "BOOL"),
    ],
)
def test_storage_empty(storage_class, shape, dtype):
    dtype = getattr(storage_class, dtype)
    storage = storage_class.empty(shape, dtype)
    assert storage.dtype == dtype
    assert storage.shape == (shape,)
    storage_data = storage.to_ndarray()
    assert storage_data.shape == (shape,)
    assert storage_data.dtype == dtype


@pytest.mark.parametrize(
    ["given", "expected"],
    [
        (
            (np.full, 5, -1, np.int32),
            (np.full, 5, -1, "INT"),
        ),
        (
            (np.zeros, 13, np.float16),
            (np.zeros, 13, "FLOAT"),
        ),
        (
            (np.ones, 11, np.bool8),
            (np.ones, 11, "BOOL"),
        ),
    ],
)
def test_storage_from_array(storage_class, given, expected):
    np_func, *args = given
    storage = storage_class.from_ndarray(np_func(*args))
    np_func, *args, dtype = expected
    dtype = getattr(storage_class, dtype)
    expected_data = np_func(*args, dtype=dtype)
    storage_data = storage.to_ndarray()
    assert storage.dtype == expected_data.dtype
    assert storage.shape == expected_data.shape
    np.testing.assert_allclose(storage_data, expected_data)


def test_storage_upload(storage_class):
    data = np.random.rand(5)
    storage = storage_class.from_ndarray(np.random.rand(5))
    storage.upload(data)
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, data)


@pytest.mark.skipif(fakeThrustRTC, reason="ThrustRTC not available")
def test_storage_urand(storage_class, random_class):
    random_gen = random_class(5, 1)
    zeros = np.zeros((4, 5), dtype=storage_class.FLOAT)
    storage = storage_class(StorageSignature(zeros.copy(), zeros.shape, zeros.dtype))
    storage.urand(random_gen)
    storage_data = storage.to_ndarray()
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, storage_data, zeros
    )


def test_storage_to_ndarray(storage_class):
    data = np.random.rand(5, 5)
    storage = storage_class.from_ndarray(data)
    np.testing.assert_allclose(storage.to_ndarray(), data)
    storage_data = storage.to_ndarray()
    assert data is not storage_data
    assert storage_data.dtype == storage_class.FLOAT
    assert storage_data.shape == (5, 5)


def test_amin(storage_class):
    data = np.arange(1, 11).reshape(5, 2)
    storage = storage_class.from_ndarray(data)
    assert storage.amin() == 1


def test_all(storage_class):
    assert storage_class.from_ndarray(
        np.full((3, 4), fill_value=True, dtype=np.bool_)
    ).all()
    assert not storage_class.from_ndarray(
        np.full((3, 5), fill_value=False, dtype=np.bool_)
    ).all()


def test_floor(storage_class):
    storage = storage_class.from_ndarray(np.asarray([-0.5, 1.25, 2.1]))
    storage = storage.floor()
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, [-1, 1, 2])


def test_floor_out_of_place(storage_class):
    storage = storage_class.from_ndarray(np.asarray([-0.5, 1.25, 2.1]))
    storage = storage.floor(storage_class.from_ndarray(np.asarray([-1.9, 0.2, 1.3])))
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, [-2, 0, 1])


def test_product(storage_class):
    data = np.asarray([1, 2, 3])
    storage = storage_class.from_ndarray(data)
    storage = storage.product(storage, storage)
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, data**2)


def test_ratio(storage_class):
    divident = np.asarray([2.4, 5.6, 3.2])
    divisor = np.asarray([9.8, 2.4, 12.3])
    storage = storage_class.from_ndarray(np.ones_like(divident))
    storage.ratio(
        storage_class.from_ndarray(divident), storage_class.from_ndarray(divisor)
    )
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, divident / divisor)


def test_divide_if_not_zero(storage_class):
    divident = np.asarray([2.4, 3.4, 3.2])
    divisor = np.asarray([9.8, 1, 12.3])
    storage = storage_class.from_ndarray(divident)
    divisor_with_0 = divisor.copy()
    divisor_with_0[1] = 0
    storage.divide_if_not_zero(storage_class.from_ndarray(divisor_with_0))
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, divident / divisor)


def test_sum(storage_class):
    a = storage_class.from_ndarray(np.ones(3))
    b = storage_class.from_ndarray(np.ones(3))
    b *= 2
    storage = storage_class.empty(3, a.dtype)
    storage.sum(a, b)
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, np.ones(3) * 3)


def test_ravel(storage_class):
    data = np.array([[1, 2, 3], [4, 5, 6]])
    storage = storage_class.empty(6, data.dtype)
    storage.ravel(data)
    storage_data = storage.to_ndarray()
    np.testing.assert_allclose(storage_data, np.ravel(data))


@pytest.mark.parametrize(
    "storage_class",
    [
        ("PySDM.storages.numba.storage", "Storage"),
    ],
    indirect=True,
)
def test_numba_storage_init(storage_class):
    data = np.ones(5, dtype=storage_class.INT)
    storage = storage_class(StorageSignature(data, data.shape, data.dtype))
    assert storage.shape == (5,)
    assert storage.dtype == storage.INT
    np.testing.assert_allclose(storage.data, data)
    assert storage.signature == StorageSignature(data, data.shape, data.dtype)
    assert len(storage) == 5


@pytest.mark.parametrize(
    ["shape_and_dtype", "expected_data"],
    [
        ((5, "INT"), np.full(5, -1, dtype=np.int64)),
        ((3, "FLOAT"), np.full(3, -1.0, dtype=np.float64)),
        ((11, "BOOL"), np.full(11, -1, dtype=np.bool_)),
    ],
)
@pytest.mark.parametrize(
    "storage_class",
    [
        ("PySDM.storages.numba.storage", "Storage"),
    ],
    indirect=True,
)
def test_numba_storage_empty(storage_class, shape_and_dtype, expected_data):
    shape, dtype = shape_and_dtype
    dtype = getattr(storage_class, dtype)
    storage = storage_class.empty(shape, dtype)
    assert storage.dtype == expected_data.dtype
    assert storage.shape == expected_data.shape
    np.testing.assert_allclose(storage.data, expected_data)


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
@pytest.mark.parametrize(
    "storage_class",
    [
        ("PySDM.storages.numba.storage", "Storage"),
    ],
    indirect=True,
)
def test_numba_storage_bool(
    storage_class, array_size, expected_context, expected_value
):
    storage = storage_class.from_ndarray(np.random.random(array_size))
    with expected_context:
        assert expected_value == bool(storage)


@pytest.mark.parametrize(
    "storage_class",
    [
        ("PySDM.storages.numba.storage", "Storage"),
    ],
    indirect=True,
)
def test_numba_storage_detach(storage_class):
    data = np.arange(1, 9, dtype=storage_class.INT)
    assert data.base is None
    sliced_data = data[3:8]
    assert sliced_data.base is not None
    storage = storage_class(
        StorageSignature(
            data=sliced_data, shape=sliced_data.shape, dtype=sliced_data.dtype
        )
    )
    assert storage.data.base is not None
    storage.detach()
    assert storage.data.base is None
