"""
clever reuser of random numbers for use in adaptive coalescence
"""

import math
from collections import namedtuple

Data = namedtuple("Data", ("data",))


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
                self.rnd.u01(self.rand)
                self.rnd.u01(self.pairs_rand)
        else:
            shift = 0
            if not hasattr(self.rnd, "JAX"):
                self.rnd.u01(self.pairs_rand)
            else:  # TODO #1913: TEMPORARY, undo this (or keep if staying with jax.random.permute)
                self.pairs_rand = Data(data=self.rnd)
            self.rnd.u01(self.rand)
        self.substep += 1
        if self.optimized_random:
            return self.pairs_rand[shift : self.particulator.n_sd + shift], self.rand
        else:
            return self.pairs_rand, self.rand
