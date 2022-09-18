"""
clever reuser of random numbers for use in adaptive coalescence
"""
import math


class RandomGeneratorOptimizer:  # pylint: disable=too-many-instance-attributes
    def __init__(self, optimized_random, dt_min, seed):
        self.particulator = None
        self.optimized_random = optimized_random
        self.dt_min = dt_min
        self.seed = seed
        self.substep = 0
        self.pairs_rand = None
        self.rand = None
        self.rnd = None

    def register(self, builder):
        self.particulator = builder.particulator
        shift = (
            math.ceil(self.particulator.dt / self.dt_min)
            if self.optimized_random
            else 0
        )
        self.pairs_rand = self.particulator.Storage.empty(
            self.particulator.n_sd + shift, dtype=float
        )
        self.rand = self.particulator.Storage.empty(
            self.particulator.n_sd // 2, dtype=float
        )
        self.rnd = self.particulator.Random(self.particulator.n_sd + shift, self.seed)

    def reset(self):
        self.substep = 0

    def get_random_arrays(self):
        if self.optimized_random:
            shift = self.substep
            if self.substep == 0:
                self.pairs_rand.urand(self.rnd)
                self.rand.urand(self.rnd)
        else:
            shift = 0
            self.pairs_rand.urand(self.rnd)
            self.rand.urand(self.rnd)
        self.substep += 1
        return self.pairs_rand[shift : self.particulator.n_sd + shift], self.rand
