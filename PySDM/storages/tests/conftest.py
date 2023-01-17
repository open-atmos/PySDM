import pytest


@pytest.fixture(
    params=[
        ("PySDM.storages.numba.storage", "Storage"),
        ("PySDM.storages.thrust_rtc.storage", "FloatStorage"),
        ("PySDM.storages.thrust_rtc.storage", "DoubleStorage"),
    ],
    ids=["numba", "thrust_rtc_float", "thrust_rtc_double"],
)
def storage_class(request):
    import importlib

    module_name, class_name = request.param
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@pytest.fixture
def random_class(storage_class):
    import importlib

    module_name = ".".join(storage_class.__module__.split(".")[:-1])
    module = importlib.import_module(f"{module_name}.random")
    return getattr(module, "Random")


@pytest.fixture
def index_backend_class(storage_class):
    import importlib

    module_name = ".".join(storage_class.__module__.split(".")[:-1])
    module = importlib.import_module(f"{module_name}.backend.index")
    return getattr(module, "IndexBackend")


@pytest.fixture
def index_class(storage_class, index_backend_class):
    from PySDM.storages.common.index import index

    return index(index_backend_class(), storage_class)


@pytest.fixture
def indexed_class(storage_class, index_class):
    from PySDM.storages.common.indexed import indexed

    return indexed(storage_class)


@pytest.fixture
def pair_backend_class(storage_class):
    import importlib

    module_name = ".".join(storage_class.__module__.split(".")[:-1])
    module = importlib.import_module(f"{module_name}.backend.pair")
    return getattr(module, "PairBackend")


@pytest.fixture
def pair_indicator_class(storage_class, pair_backend_class):
    from PySDM.storages.common.pair_indicator import pair_indicator

    return pair_indicator(pair_backend_class(), storage_class)


@pytest.fixture
def pairwise_class(storage_class, pair_backend_class):
    from PySDM.storages.common.pairwise import pairwise

    return pairwise(pair_backend_class(), storage_class)
