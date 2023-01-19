import os

import numpy as np
import pytest

from PySDM.storages.thrust_rtc.test_helpers.flag import fakeThrustRTC


@pytest.fixture()
def pair_indicator_instance(storage_class, pair_indicator_class):
    # Arrange
    pair_indicator_instance_ = pair_indicator_class(6)
    pair_indicator_data = np.ones(6)
    pair_indicator_data[np.arange(1, 6, 2)] = 0
    pair_indicator_instance_.indicator = storage_class.from_ndarray(
        pair_indicator_data.copy()
    )
    return pair_indicator_instance_


@pytest.fixture()
def pairwise_storage_instance(storage_class, pairwise_class):
    return pairwise_class.empty(3, storage_class.FLOAT)


@pytest.mark.skipif(fakeThrustRTC, reason="ThrustRTC not available")
def test_distance(
    pair_indicator_instance, pairwise_storage_instance, indexed_class, index_class
):
    data = np.array([1, 4, 2, 5, 3, 6])
    data_idx = index_class.identity_index(6)
    pairwise_storage_instance.distance(
        indexed_class.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(
        pairwise_storage_instance.to_ndarray(), np.asarray([3, 3, 3])
    )


@pytest.mark.skipif(fakeThrustRTC, reason="ThrustRTC not available")
def test_max(
    pair_indicator_instance, pairwise_storage_instance, indexed_class, index_class
):
    data = np.array([1, -1, 2, 4, 3, 3])
    data_idx = index_class.identity_index(6)
    pairwise_storage_instance.max(
        indexed_class.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(
        pairwise_storage_instance.to_ndarray(), np.asarray([1, 4, 3])
    )


@pytest.mark.skipif(fakeThrustRTC, reason="ThrustRTC not available")
def test_min(
    pair_indicator_instance, pairwise_storage_instance, indexed_class, index_class
):
    data = np.array([1, -1, 2, 4, 3, 3])
    data_idx = index_class.identity_index(6)
    pairwise_storage_instance.min(
        indexed_class.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(
        pairwise_storage_instance.to_ndarray(), np.asarray([-1, 2, 3])
    )


@pytest.mark.skipif(fakeThrustRTC, reason="ThrustRTC not available")
def test_multiply(
    pair_indicator_instance, pairwise_storage_instance, indexed_class, index_class
):
    data = np.array([1, -1, 2, 4, 3, 3])
    data_idx = index_class.identity_index(6)
    pairwise_storage_instance.multiply(
        indexed_class.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(
        pairwise_storage_instance.to_ndarray(), np.asarray([-1, 8, 9])
    )


@pytest.mark.skipif(fakeThrustRTC, reason="ThrustRTC not available")
def test_sum(
    pair_indicator_instance, pairwise_storage_instance, indexed_class, index_class
):
    data = np.array([1, -1, 2, 4, 3, 3])
    data_idx = index_class.identity_index(6)
    pairwise_storage_instance.sum(
        indexed_class.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(
        pairwise_storage_instance.to_ndarray(), np.asarray([0, 6, 6])
    )


@pytest.mark.skipif(fakeThrustRTC, reason="ThrustRTC not available")
def test_sort(
    pair_indicator_instance,
    pairwise_storage_instance,
    indexed_class,
    index_class,
    pairwise_class,
    storage_class,
):
    # Arrange
    pairwise_storage_instance = pairwise_class.empty(6, storage_class.FLOAT)
    data = np.array([-1, 1, 4, 2, 3, 3])
    data_idx = index_class.identity_index(6)
    pairwise_storage_instance.sort(
        indexed_class.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(
        pairwise_storage_instance.to_ndarray(), np.asarray([1, -1, 4, 2, 3, 3])
    )


@pytest.mark.parametrize(
    "storage_class",
    [
        ("PySDM.storages.numba.storage", "Storage"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "_data_in, _data_out, _is_first_in_pair, _idx",
    (
        pytest.param(
            [44.0, 666.0],
            [
                0,
            ],
            [True, True],
            [0, 1],
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            [44.0, 666.0],
            [
                0,
            ],
            [True, False],
            [0, 1],
        ),
    ),
)
# pylint: disable=redefined-outer-name
def test_sum_pair_body_out_of_bounds(
    _data_in,
    _data_out,
    _is_first_in_pair,
    _idx,
    storage_class,
    pair_indicator_class,
    pair_backend_class,
):
    # Arrange
    data_out = storage_class.from_ndarray(np.asarray(_data_out))
    data_in = storage_class.from_ndarray(np.asarray(_data_in))

    is_first_in_pair = pair_indicator_class(len(_is_first_in_pair))
    is_first_in_pair.indicator = storage_class.from_ndarray(
        np.asarray(_is_first_in_pair)
    )

    idx = storage_class.from_ndarray(np.asarray(_idx))
    pair_backend = pair_backend_class()
    sut = (
        pair_backend.sum_pair_body
        if "NUMBA_DISABLE_JIT" in os.environ
        else pair_backend.sum_pair_body.py_func
    )
    # Act
    sut(
        data_out.data,
        data_in.data,
        is_first_in_pair.indicator.data,
        idx.data,
        len(idx),
    )

    # Assert


@pytest.mark.parametrize(
    "_data_in, _data_out, _idx",
    (
        pytest.param(
            [44.0, 666.0],
            [
                0,
            ],
            [0, 1],
        ),
    ),
)
def test_sum_pair(
    storage_class, pair_indicator_class, pair_backend_class, _data_in, _data_out, _idx
):
    # Arrange

    data_out = storage_class.from_ndarray(np.asarray(_data_out))
    data_in = storage_class.from_ndarray(np.asarray(_data_in))
    is_first_in_pair = pair_indicator_class(len(_data_in))
    is_first_in_pair.indicator = storage_class.from_ndarray(
        np.asarray(
            [True, False],
        )
    )

    idx = storage_class.from_ndarray(np.asarray(_idx))

    # Act
    pair_backend_class().sum_pair(data_out, data_in, is_first_in_pair, idx)

    # Assert
    np.testing.assert_array_equal(data_out, [44.0 + 666.0])


@pytest.mark.parametrize("length", (1, 2, 3, 4))
def test_find_pairs_length(
    storage_class, pair_indicator_class, index_class, pair_backend_class, length
):
    # arrange
    n_sd = 4

    cell_start = storage_class.from_ndarray(np.asarray([0, 0, 0, 0]))
    cell_id = storage_class.from_ndarray(np.asarray([0, 0, 0, 0]))
    cell_idx = storage_class.from_ndarray(np.asarray([0, 1, 2, 3]))
    is_first_in_pair = pair_indicator_class(n_sd)
    is_first_in_pair.indicator = storage_class.from_ndarray(np.asarray([True] * n_sd))
    idx = index_class.identity_index(n_sd)

    # act
    idx.length = length
    pair_backend_class().find_pairs(
        cell_start, is_first_in_pair, cell_id, cell_idx, idx
    )

    # assert
    assert not is_first_in_pair.indicator[length - 1]
