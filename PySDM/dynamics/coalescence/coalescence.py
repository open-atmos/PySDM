"""
Created at 07.06.2019
"""

import numpy as np

from .random_generator_optimizer import RandomGeneratorOptimizer


class Coalescence:

    def __init__(self, kernel, seed=None, croupier=None, adaptive=False, max_substeps=128, optimized_random=False):
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

        self.actual_length = None

    def register(self, builder):
        self.core = builder.core
        self.temp = self.core.PairwiseStorage.empty(self.core.n_sd, dtype=float)
        self.prob = self.core.PairwiseStorage.empty(self.core.n_sd, dtype=float)
        self.is_first_in_pair = self.core.PairIndicator(self.core.n_sd)
        self.n_substep = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.n_substep[:] = 1
        self.rnd_opt.register(builder)
        self.kernel.register(builder)

        self.adaptive_memory = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.subs = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.msub = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

        if self.croupier is None:
            self.croupier = self.core.backend.default_croupier

        self.collision_rate = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.collision_rate_deficit = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

    @property
    def max_substeps(self):
        return self.rnd_opt.max_substeps

    def __call__(self):
        # TODO dt
        if self.enable:
            if not self.adaptive:
                self.step(0, self.adaptive_memory)
            else:
                self.actual_length = self.core.particles._Particles__idx.length
                self.core.particles.cell_idx.data = self.n_substep.data.argsort(kind="stable")[::-1]
                for s in range(max(self.n_substep.data)):  # range(self.n_substep[0]):
                    self.step(s, self.adaptive_memory)
                    self.subs[:] += self.adaptive_memory
                    method1(self.adaptive_memory.data, self.msub.data)

                self.core.particles._Particles__idx.length = self.actual_length

                method2(self.n_substep.data, self.msub.data, self.max_substeps, self.subs.data)
                self.subs[:] = 0
                self.msub[:] = 0

                self.core.particles.cell_idx.data = np.arange(len(self.core.particles.cell_idx.data)).astype(
                    dtype=np.int64)  # TODO
                self.core.particles._Particles__sort_by_cell_id()

    def step(self, s, adaptive_memory):
        pairs_rand, rand = self.rnd_opt.get_random_arrays(s)
        self.toss_pairs(self.is_first_in_pair, pairs_rand, s)
        self.compute_probability(self.prob, self.is_first_in_pair)
        self.compute_gamma(self.prob, rand)
        if self.adaptive:
            adaptive_memory[:] = 1
        self.core.particles.coalescence(gamma=self.prob, adaptive=self.adaptive, subs=self.n_substep,
                                        adaptive_memory=adaptive_memory,
                                        collision_rate=self.collision_rate,
                                        collision_rate_deficit=self.collision_rate_deficit)

    def toss_pairs(self, is_first_in_pair, u01, s):
        if self.adaptive:
            self.core.particles._Particles__idx.length = self.actual_length
        self.core.particles.sanitize()
        if self.adaptive:
            self.actual_length = self.core.particles._Particles__idx.length

        self.core.particles.permutation(u01, self.croupier == 'local')

        if self.adaptive:
            end = method3(self.n_substep.data, self.core.mesh.n_cell, self.core.particles.cell_start.data,
                          self.core.particles.cell_idx.data, s)
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


import numba


@numba.njit()
def method1(adaptive_memory, msub):
    for i in range(len(adaptive_memory)):
        msub[i] = max(msub[i], adaptive_memory[i])


@numba.njit()
def method2(n_substep, msub, max_substeps, subs):
    for i in range(len(n_substep)):
        n_substep[i] = min(max_substeps, ((subs[i] / n_substep[i]) + msub[i]) // 2)


@numba.njit()
def method3(n_substep, n_cell, cell_start, cell_idx, s):
    end = 0
    for i in range(n_cell - 1, -1, -1):
        if n_substep[i] <= s:
            continue
        else:
            end = cell_start[cell_idx[i] + 1]
            break
    return end
