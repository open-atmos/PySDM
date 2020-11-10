"""
Created at 10.11.2020
"""


class RandomGeneratorOptimizer:

    def __init__(self, core, optimized_random, max_substeps, seed):
        self.core = core
        self.optimized_random = optimized_random
        shift = max_substeps if optimized_random else 0
        self.pairs_rand = core.Storage.empty(core.n_sd + shift, dtype=float)
        self.rand = core.Storage.empty(core.n_sd // 2, dtype=float)
        self.rnd = core.Random(core.n_sd + shift, seed)

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
