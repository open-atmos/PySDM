"""
TODO #744
"""

import math


class RandomGeneratorOptimizerNoPair:
    def __init__(self, optimized_random, dt_min, seed):
        self.particulator = None
        self.optimized_random = optimized_random
        self.dt_min = dt_min
        self.seed = seed
        self.substep = 0
        self.rand = None
        self.rnd = None

    def register(self, builder):
        self.particulator = builder.particulator
        shift = (
            math.ceil(self.particulator.dt / self.dt_min)
            if self.optimized_random
            else 0
        )
        self.rand = self.particulator.Storage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.rnd = self.particulator.Random(self.particulator.n_sd + shift, self.seed)

    def reset(self):
        self.substep = 0

    def get_random_arrays(self):
        if self.optimized_random:
            if self.substep == 0:
                self.rand.urand(self.rnd)
        else:
            self.rand.urand(self.rnd)
        self.substep += 1
        return self.rand
