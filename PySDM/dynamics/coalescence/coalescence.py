"""
Created at 07.06.2019
"""

import numpy as np

from .random_generator_optimizer import RandomGeneratorOptimizer


class Coalescence:

    def __init__(self, kernel, seed=None, croupier='local', adaptive=True, max_substeps=128, optimized_random=False):
        self.core = None
        self.kernel = kernel
        self.rnd_opt = RandomGeneratorOptimizer(optimized_random=optimized_random, max_substeps=max_substeps, seed=seed)
        self.enable = True
        self.adaptive = adaptive
        self.n_substep = None
        self.croupier = croupier

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
        # TODO dt
        if self.enable:
            adaptive_memory = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
            subs = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
            msub = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
            length_cache = self.core.particles._Particles__idx.length
            for s in range(self.n_substep[0]):
                self.step(s, adaptive_memory)
                subs[:] += adaptive_memory
                for i in range(len(adaptive_memory)):
                    msub.data[i] = max(msub.data[i], adaptive_memory.data[i])
            self.core.particles._Particles__idx.length = length_cache

            if self.adaptive:
                for i in range(len(self.n_substep)):
                    self.n_substep[i] = min(self.max_substeps, int(((subs[i] / self.n_substep[i]) + msub[i]) / 2))

    def step(self, s, adaptive_memory):
        pairs_rand, rand = self.rnd_opt.get_random_arrays(s)
        self.toss_pairs(self.is_first_in_pair, pairs_rand, s)
        self.compute_probability(self.prob, self.is_first_in_pair)
        self.compute_gamma(self.prob, rand)
        adaptive_memory[:] = 1
        self.core.particles.coalescence(gamma=self.prob, adaptive=self.adaptive, subs=self.n_substep,
                                        adaptive_memory=adaptive_memory)

    def toss_pairs(self, is_first_in_pair, u01, s):
        self.core.particles.sanitize()
        self.core.particles.permutation(u01, self.croupier == 'local')
        if s == 0:
            self.core.particles.cell_idx.data = self.n_substep.data.argsort(kind="stable")[::-1]
        end = 0
        for i in range(self.core.mesh.n_cell - 1, -1, -1):
            if self.n_substep.data[i] < s:
                continue
            else:
                end = self.core.particles.cell_start[self.core.particles.cell_idx.data[i] + 1]
        self.core.particles._Particles__idx.length = end
        is_first_in_pair.update(
            self.core.particles.cell_start,
            self.core.particles.cell_idx,
            self.core.particles['cell id']
        )

    def compute_probability(self, prob, is_first_in_pair):
        self.kernel(self.temp, is_first_in_pair)
        prob.max(self.core.particles['n'], is_first_in_pair)
        prob *= self.temp

        norm_factor = self.temp
        self.core.normalize(prob, norm_factor, self.n_substep)

    def compute_gamma(self, prob, rand):
        self.core.backend.compute_gamma(prob, rand)
