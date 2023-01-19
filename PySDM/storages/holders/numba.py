from PySDM.storages.common.index import index
from PySDM.storages.common.indexed import indexed
from PySDM.storages.common.pair_indicator import pair_indicator
from PySDM.storages.common.pairwise import pairwise
from PySDM.storages.common.storage import StoragesHolder
from PySDM.storages.numba.backend.index import IndexBackend
from PySDM.storages.numba.backend.pair import PairBackend
from PySDM.storages.numba.random import Random as NumbaRandom
from PySDM.storages.numba.storage import Storage as NumbaStorage


class NumbaStorageHolder(StoragesHolder):
    Storage = NumbaStorage
    Random = NumbaRandom
    Index = index(IndexBackend(), Storage)
    IndexedStorage = indexed(Storage)
    PairwiseStorage = pairwise(PairBackend(), Storage)
    PairIndicator = pair_indicator(PairBackend(), Storage)
