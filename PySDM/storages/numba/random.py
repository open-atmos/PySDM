"""
random number generator class for Numba backend
"""
import numpy as np

from PySDM.storages.common.random import Random as BaseRandom


class Random(BaseRandom):  # pylint: disable=too-few-public-methods
    def __init__(self, size, seed):
        super().__init__(size, seed)
        self.generator = np.random.default_rng(seed)

    def __call__(self, storage):
        storage.data[:] = self.generator.uniform(0, 1, storage.shape)
