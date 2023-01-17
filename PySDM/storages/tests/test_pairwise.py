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
