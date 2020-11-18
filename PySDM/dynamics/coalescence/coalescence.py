"""
Created at 07.06.2019
"""

from .random_generator_optimizer import RandomGeneratorOptimizer


class Coalescence:

    def __init__(self, kernel, seed=None, max_substeps=128, optimized_random=False):
        self.core = None
        self.kernel = kernel
        self.rnd_opt = RandomGeneratorOptimizer(optimized_random=optimized_random, max_substeps=max_substeps, seed=seed)
        self.enable = True
        self.adaptive = False
        self.n_substep = None
        self.croupier = 'local'

        self.temp = None
        self.prob = None
        self.is_first_in_pair = None

    def register(self, builder):
        self.core = builder.core
        self.temp = self.core.PairwiseStorage.empty(self.core.n_sd, dtype=float)
        self.prob = self.core.PairwiseStorage.empty(self.core.n_sd, dtype=float)
        self.is_first_in_pair = self.core.PairIndicator(self.core.n_sd)
        self.n_substep = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.n_substep[:] = 1
        self.rnd_opt.register(builder)
        self.kernel.register(builder)

    @property
    def max_substeps(self):
        return self.rnd_opt.max_substeps

    def __call__(self):
        if self.enable:
            subs = 0
            msub = 1
            for s in range(self.n_substep[0]):
                sub = self.step(s)
                subs += sub
                msub = max(msub, sub)

            if self.adaptive:
                self.n_substep = min(self.max_substeps, int(((subs / self.n_substep) + msub) / 2))

    def step(self, s):
        pairs_rand, rand = self.rnd_opt.get_random_arrays(s)
        self.toss_pairs(self.is_first_in_pair, pairs_rand)
        self.compute_probability(self.prob, self.is_first_in_pair, self.n_substep)
        self.compute_gamma(self.prob, rand)
        sub = self.core.coalescence(gamma=self.prob, adaptive=self.adaptive, subs=self.n_substep,
                                    adaptive_memory=self.temp)
        return sub

    def toss_pairs(self, is_first_in_pair, u01):
        self.core.particles.sanitize()
        self.core.particles.permutation(u01, self.croupier == 'local')
        is_first_in_pair.update(
            self.core.particles.cell_start,
            self.core.particles.cell_idx,
            self.core.particles['cell id']
        )

    def compute_probability(self, prob, is_first_in_pair, subs):
        self.kernel(self.temp, is_first_in_pair)
        prob.max(self.core.particles['n'], is_first_in_pair)
        prob *= self.temp

        norm_factor = self.temp
        self.core.normalize(prob, norm_factor, subs)

    def compute_gamma(self, prob, rand):
        self.core.backend.compute_gamma(prob, rand)
