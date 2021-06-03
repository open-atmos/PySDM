"""
Created at 27.05.2021
"""

import math


class RandomGeneratorOptimizerNoPair:

    def __init__(self, optimized_random, dt_min, seed):
        self.core = None
        self.optimized_random = optimized_random
        self.dt_min = dt_min
        self.seed = seed
        self.substep = 0
        self.rand = None
        self.rnd = None

    def register(self, builder):
        self.core = builder.core
        shift = math.ceil(self.core.dt / self.dt_min) if self.optimized_random else 0
        self.rand = self.core.Storage.empty(self.core.n_sd // 2, dtype=float)
        self.rnd = self.core.Random(self.core.n_sd + shift, self.seed)

    def reset(self):
        self.substep = 0

    def get_random_arrays(self):
        if self.optimized_random:
            shift = self.substep
            if self.substep == 0:
                self.rand.urand(self.rnd)
        else:
            shift = 0
            self.rand.urand(self.rnd)
        self.substep += 1
        return self.rand
