"""
Created at 07.06.2019
"""

from .random_generator_optimizer import RandomGeneratorOptimizer


class Coalescence:

    def __init__(self, kernel, seed=None, max_substeps=128):
        self.core = None
        self.kernel = kernel
        self.enable = True
        self.stats_steps = 0
        self.adaptive = False
        self.optimized_random = True
        self.max_substeps = max_substeps
        self.subs = 1
        self.croupier = 'local'
        self.seed = seed

        self.temp = None
        self.rnd_opt = None
        self.prob = None
        self.is_first_in_pair = None
        self.rnd = None

    def register(self, builder):
        self.core = builder.core
        self.temp = self.core.IndexedStorage.empty(self.core.n_sd, dtype=float)
        self.rnd_opt = RandomGeneratorOptimizer(self.core, self.optimized_random, self.max_substeps, self.seed)
        self.prob = self.core.IndexedStorage.empty(self.core.n_sd, dtype=float)
        self.is_first_in_pair = self.core.IndexedStorage.empty(self.core.n_sd, dtype=int)  # TODO bool
        self.kernel.register(builder)

    def __call__(self):
        if self.enable:
            subs = 0
            msub = 1
            for s in range(self.subs):
                pairs_rand, rand = self.rnd_opt.get_random_arrays(s)
                self.toss_pairs(self.is_first_in_pair, pairs_rand)
                self.compute_probability(self.prob, self.is_first_in_pair, self.subs)
                sub = self.coalescence(self.prob, rand, self.adaptive, self.subs)
                subs += sub
                msub = max(msub, sub)

            self.stats_steps += self.subs
            if self.adaptive:
                self.subs = min(self.max_substeps, int(((subs/self.subs) + msub)/2))

    def compute_gamma(self, prob, rand):
        self.core.backend.compute_gamma(prob, rand)

    def coalescence(self, prob, rand, adaptive, subs):
        self.compute_gamma(prob, rand)
        return self.core.coalescence(gamma=prob, adaptive=adaptive, subs=subs, adaptive_memory=self.temp)

    def compute_probability(self, prob, is_first_in_pair, subs):
        self.kernel(self.temp, is_first_in_pair)
        prob.max_pair(self.core.particles['n'], is_first_in_pair)
        prob *= self.temp

        norm_factor = self.temp
        self.core.normalize(prob, norm_factor, subs)

    def toss_pairs(self, is_first_in_pair, u01):
        self.core.particles.sanitize()
        self.core.particles.permutation(u01, self.croupier == 'local')
        is_first_in_pair.find_pairs(self.core.particles.cell_start, self.core.particles['cell id'])
