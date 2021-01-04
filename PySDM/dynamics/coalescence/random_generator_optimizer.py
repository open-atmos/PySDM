"""
Created at 10.11.2020
"""


class RandomGeneratorOptimizer:

    def __init__(self, optimized_random, max_substeps, seed):
        self.core = None
        self.optimized_random = optimized_random
        self.max_substeps = max_substeps
        self.seed = seed
        self.pairs_rand = None
        self.rand = None
        self.rnd = None

    def register(self, builder):
        self.core = builder.core
        shift = self.max_substeps if self.optimized_random else 0
        self.pairs_rand = self.core.Storage.empty(self.core.n_sd + shift, dtype=float)
        self.rand = self.core.Storage.empty(self.core.n_sd // 2, dtype=float)
        self.rnd = self.core.Random(self.core.n_sd + shift, self.seed)

    def get_random_arrays(self, s):
        shift = 0
        if self.optimized_random:
            shift = s
            if s == 0:
                self.pairs_rand.urand(self.rnd)
                self.rand.urand(self.rnd)
        else:
            self.pairs_rand.urand(self.rnd)
            self.rand.urand(self.rnd)
        return self.pairs_rand[shift:self.core.n_sd + shift], self.rand
