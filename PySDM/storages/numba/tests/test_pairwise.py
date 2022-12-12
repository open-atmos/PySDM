import numpy as np
import pytest

from PySDM.storages.common.index import index
from PySDM.storages.common.indexed import indexed
from PySDM.storages.common.pair_indicator import pair_indicator
from PySDM.storages.common.pairwise import pairwise
from PySDM.storages.numba.backend.index import IndexBackend
from PySDM.storages.numba.backend.pair import PairBackend
from PySDM.storages.numba.storage import Storage


@pytest.fixture()
def index_backend():
    return IndexBackend()


@pytest.fixture()
def index_cls(index_backend):
    return index(index_backend, storage_cls=Storage)


@pytest.fixture()
def pair_backend():
    return PairBackend()


@pytest.fixture()
def pair_indicator_cls(pair_backend):
    return pair_indicator(pair_backend, storage_cls=Storage)


@pytest.fixture()
def pairwise_storage_cls(pair_backend):
    return pairwise(pair_backend, storage_cls=Storage)


@pytest.fixture()
def indexed_storage_cls():
    return indexed(Storage)


@pytest.fixture()
def pair_indicator_instance(pair_indicator_cls):
    # Arrange
    pair_indicator_instance_ = pair_indicator_cls(6)
    pair_indicator_data = np.ones(6)
    pair_indicator_data[np.arange(1, 6, 2)] = 0
    pair_indicator_instance_.indicator.data = pair_indicator_data.copy()
    return pair_indicator_instance_


@pytest.fixture()
def pairwise_storage_instance(pairwise_storage_cls):
    return pairwise_storage_cls.empty(3, Storage.FLOAT)


def test_distance(
    pair_indicator_instance, pairwise_storage_instance, indexed_storage_cls, index_cls
):
    data = np.array([1, 4, 2, 5, 3, 6])
    data_idx = index_cls.identity_index(6)
    pairwise_storage_instance.distance(
        indexed_storage_cls.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(pairwise_storage_instance.data, np.asarray([3, 3, 3]))


def test_max(
    pair_indicator_instance, pairwise_storage_instance, indexed_storage_cls, index_cls
):
    data = np.array([1, -1, 2, 4, 3, 3])
    data_idx = index_cls.identity_index(6)
    pairwise_storage_instance.max(
        indexed_storage_cls.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(pairwise_storage_instance.data, np.asarray([1, 4, 3]))


def test_min(
    pair_indicator_instance, pairwise_storage_instance, indexed_storage_cls, index_cls
):
    data = np.array([1, -1, 2, 4, 3, 3])
    data_idx = index_cls.identity_index(6)
    pairwise_storage_instance.min(
        indexed_storage_cls.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(pairwise_storage_instance.data, np.asarray([-1, 2, 3]))


def test_multiply(
    pair_indicator_instance, pairwise_storage_instance, indexed_storage_cls, index_cls
):
    data = np.array([1, -1, 2, 4, 3, 3])
    data_idx = index_cls.identity_index(6)
    pairwise_storage_instance.multiply(
        indexed_storage_cls.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(pairwise_storage_instance.data, np.asarray([-1, 8, 9]))


def test_sum(
    pair_indicator_instance, pairwise_storage_instance, indexed_storage_cls, index_cls
):
    data = np.array([1, -1, 2, 4, 3, 3])
    data_idx = index_cls.identity_index(6)
    pairwise_storage_instance.sum(
        indexed_storage_cls.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(pairwise_storage_instance.data, np.asarray([0, 6, 6]))


def test_sort(
    pair_indicator_instance, indexed_storage_cls, pairwise_storage_cls, index_cls
):
    # Arrange
    pairwise_storage_instance = pairwise_storage_cls.empty(6, Storage.FLOAT)
    data = np.array([-1, 1, 4, 2, 3, 3])
    data_idx = index_cls.identity_index(6)
    pairwise_storage_instance.sort(
        indexed_storage_cls.indexed_from_ndarray(data_idx, data.copy()),
        pair_indicator_instance,
    )
    np.testing.assert_allclose(
        pairwise_storage_instance.data, np.asarray([1, -1, 4, 2, 3, 3])
    )
