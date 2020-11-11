"""
Created at 07.06.2019
"""

from .random_generator_optimizer import RandomGeneratorOptimizer


class Coalescence:

    def __init__(self, kernel, seed=None, max_substeps=128):
        self.core = None
        self.kernel = kernel
        self.rnd_opt = RandomGeneratorOptimizer(optimized_random=True, max_substeps=max_substeps, seed=seed)
        self.enable = True
        self.adaptive = False
        self.substep_num = 1
        self.croupier = 'local'

        self.temp = None
        self.prob = None
        self.is_first_in_pair = None

    @property
    def max_substeps(self):
        return self.rnd_opt.max_substeps

    def register(self, builder):
        self.core = builder.core
        self.temp = self.core.IndexedStorage.empty(self.core.n_sd, dtype=float)
        self.prob = self.core.IndexedStorage.empty(self.core.n_sd, dtype=float)
        self.is_first_in_pair = self.core.IndexedStorage.empty(self.core.n_sd, dtype=int)  # TODO bool
        self.rnd_opt.register(builder)
        self.kernel.register(builder)

    def __call__(self):
        if self.enable:
            subs = 0
            msub = 1
            for s in range(self.substep_num):
                sub = self.step(s)
                subs += sub
                msub = max(msub, sub)

            if self.adaptive:
                self.substep_num = min(self.max_substeps, int(((subs / self.substep_num) + msub) / 2))

    def step(self, s):
        pairs_rand, rand = self.rnd_opt.get_random_arrays(s)
        self.toss_pairs(self.is_first_in_pair, pairs_rand)
        self.compute_probability(self.prob, self.is_first_in_pair, self.substep_num)
        self.compute_gamma(self.prob, rand)
        sub = self.core.coalescence(gamma=self.prob, adaptive=self.adaptive, subs=self.substep_num,
                                    adaptive_memory=self.temp)
        return sub

    def toss_pairs(self, is_first_in_pair, u01):
        self.core.particles.sanitize()
        self.core.particles.permutation(u01, self.croupier == 'local')
        is_first_in_pair.find_pairs(self.core.particles.cell_start, self.core.particles['cell id'])

    def compute_probability(self, prob, is_first_in_pair, subs):
        self.kernel(self.temp, is_first_in_pair)
        prob.max_pair(self.core.particles['n'], is_first_in_pair)
        prob *= self.temp

        norm_factor = self.temp
        self.core.normalize(prob, norm_factor, subs)

    def compute_gamma(self, prob, rand):
        self.core.backend.compute_gamma(prob, rand)
