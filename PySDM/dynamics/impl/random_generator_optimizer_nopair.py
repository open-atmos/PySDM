"""
TODO #744
"""

import math


class RandomGeneratorThingNoPair:
    def __init__(self, dt_min, seed):
        self.particulator = None
        self.dt_min = dt_min
        self.seed = seed
        self.rand = None
        self.rnd = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.rand = self.particulator.Storage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.rnd = self.particulator.Random(self.particulator.n_sd, self.seed)

    def get_random_arrays(self):
        self.rand.urand(self.rnd)
        return self.rand
