import numpy as np

from PySDM.storages.common.index import index
from PySDM.storages.common.indexed import indexed
from PySDM.storages.common.pair_indicator import pair_indicator
from PySDM.storages.common.pairwise import pairwise
from PySDM.storages.common.storage import StoragesHolder
from PySDM.storages.thrust_rtc.backend.index import IndexBackend
from PySDM.storages.thrust_rtc.backend.pair import PairBackend
from PySDM.storages.thrust_rtc.conf import trtc
from PySDM.storages.thrust_rtc.random import Random as ThrustRTC_Random
from PySDM.storages.thrust_rtc.storage import DoubleStorage, FloatStorage


class ThrustRTCStorageHolder(StoragesHolder):
    Random = ThrustRTC_Random

    def __init__(self, double_precision=False):
        self._conv_function = trtc.DVDouble if double_precision else trtc.DVFloat
        self._real_type = "double" if double_precision else "float"
        self._np_dtype = np.float64 if double_precision else np.float32

        self.Storage = DoubleStorage if double_precision else FloatStorage
        self.Index = index(IndexBackend(), self.Storage)
        self.IndexedStorage = indexed(self.Storage)
        self.PairwiseStorage = pairwise(PairBackend(), self.Storage)
        self.PairIndicator = pair_indicator(PairBackend(), self.Storage)
