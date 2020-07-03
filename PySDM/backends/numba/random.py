"""
Created at 03.07.2020
"""

import numpy as np

from PySDM.backends.numba.incrementation import Incrementation


#  TIP: can be call asynchronous
#  TIP: sometimes only halve array is needed

class Random:
    def __init__(self, size, seed=None):
        self.size = size
        seed = seed or np.random.randint(0, 2*16)
        self.seed = Incrementation(seed)

    def __call__(self, storage):
        np.random.seed(self.seed())
        storage.data[:] = np.random.uniform(0, 1, storage.shape)

