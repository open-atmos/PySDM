"""
Created at 03.07.2020
"""

import numpy as np


#  TIP: can be call asynchronous
#  TIP: sometimes only half array is needed

class Random:
    def __init__(self, size, seed=None):
        self.size = size
        seed = seed or np.random.randint(0, 2*16)
        self.generator = np.random.default_rng(seed)

    def __call__(self, storage):
        storage.data[:] = self.generator.uniform(0, 1, storage.shape)

