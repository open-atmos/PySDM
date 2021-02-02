"""
Created at 07.06.2019
"""

import numpy as np
from PySDM.physics import si
from .random_generator_optimizer import RandomGeneratorOptimizer

default_dt_coal_range = (1 * si.second, 5 * si.second)


class Coalescence:

    def __init__(self,
                 kernel,
                 seed=None,
                 croupier=None,
                 optimized_random=False,
                 substeps: int = 1,
                 adaptive: bool = False,
                 dt_coal_range=default_dt_coal_range
                 ):

        self.core = None
        self.enable = True

        self.kernel = kernel

        self.rnd_opt = RandomGeneratorOptimizer(optimized_random=optimized_random,
                                                dt_min=dt_coal_range[0],
                                                seed=seed)
        self.croupier = croupier

        self.__substeps = substeps
        self.adaptive = adaptive
        self.n_substep = None
        self.dt_coal_range = dt_coal_range

        self.kernel_temp = None
        self.prob = None
        self.is_first_in_pair = None
        self.remaining_dt = None

        self.actual_length = None

    def register(self, builder):
        self.core = builder.core

        if self.adaptive:
            assert self.core.dt >= self.dt_coal_range[0]
            assert self.core.dt >= self.dt_coal_range[1]

        self.kernel_temp = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.norm_factor_temp = self.core.Storage.empty(self.core.n_sd, dtype=float)  # TODO #372
        self.prob = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.is_first_in_pair = self.core.PairIndicator(self.core.n_sd)
        self.remaining_dt = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)

        self.n_substep = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.n_substep[:] = self.__substeps

        self.rnd_opt.register(builder)
        self.kernel.register(builder)

        self.adaptive_memory = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.subs = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.msub = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

        if self.croupier is None:
            self.croupier = self.core.backend.default_croupier

        self.collision_rate = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.collision_rate_deficit = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

    def __call__(self):
        if self.enable:
            if not self.adaptive:
                self.step(0, self.adaptive_memory)
            else:
                self.remaining_dt[:] = self.core.dt
                self.actual_length = self.core.particles._Particles__idx.length
                self.actual_cell_idx = self.core.particles.cell_idx.data
                self.core.particles.cell_idx.data = self.remaining_dt.data.argsort(kind="stable")[::-1]

                # end = self.actual_length
                s = 0
                while self.core.particles._Particles__idx.length != 0:
                    self.step(s, self.adaptive_memory)
                    s += 1

                    # self.subs[:] += self.adaptive_memory
                    # method1(self.adaptive_memory.data, self.msub.data)

                self.core.particles._Particles__idx.length = self.actual_length

                # method2(self.n_substep.data, self.msub.data, int(self.core.dt / self.dt_coal_range[0]), int(self.core.dt / self.dt_coal_range[1]), self.subs.data)
                # self.subs[:] = 0
                # self.msub[:] = 0

                self.core.particles.cell_idx.data = self.actual_cell_idx
                self.core.particles._Particles__sort_by_cell_id()

    def step(self, s, adaptive_memory):
        pairs_rand, rand = self.rnd_opt.get_random_arrays(s)
        self.toss_pairs(self.is_first_in_pair, pairs_rand, s)
        self.compute_probability(self.prob, self.is_first_in_pair)
        self.compute_gamma(self.prob, rand)
        if self.adaptive:
            adaptive_memory[:] = 1
        self.core.particles.coalescence(gamma=self.prob)

    def toss_pairs(self, is_first_in_pair, u01, s):
        if self.adaptive:
            self.core.particles._Particles__idx.length = self.actual_length
        self.core.particles.sanitize()
        if self.adaptive:
            self.actual_length = self.core.particles._Particles__idx.length

        self.core.particles.permutation(u01, self.croupier == 'local')

        if self.adaptive:
            end = method3(self.remaining_dt.data, self.core.mesh.n_cell, self.core.particles.cell_start.data, s)
            self.core.particles._Particles__idx.length = end

        is_first_in_pair.update(
            self.core.particles.cell_start,
            self.core.particles.cell_idx,
            self.core.particles['cell id']
        )

        self.core.particles.sort_within_pair_by_attr(is_first_in_pair, attr_name="n")

    def compute_probability(self, prob, is_first_in_pair):
        self.kernel(self.kernel_temp, is_first_in_pair)
        prob.max(self.core.particles['n'], is_first_in_pair)
        prob *= self.kernel_temp

        self.core.normalize(prob, self.norm_factor_temp)

    def compute_gamma(self, prob, rand):
        self.core.backend.compute_gamma(prob, rand, self.core.particles._Particles__idx, self.core.particles['n'],
                                        self.adaptive, self.adaptive_memory, self.core.particles["cell id"], self.subs,
                                        self.collision_rate_deficit, self.collision_rate, self.remaining_dt, self.core.dt)


# TODO #69
import numba


# @numba.njit()
# def method1(adaptive_memory, msub):
#     for i in range(len(adaptive_memory)):
#         msub[i] = max(msub[i], adaptive_memory[i])
#
#
# @numba.njit()
# def method2(n_substep, msub, max_substeps, min_substeps, subs):
#     for i in range(len(n_substep)):
#         n_substep[i] = min(max_substeps, msub[i])#TODO ((subs[i] / n_substep[i]) + msub[i]) // 2)
#         # n_substep[i] = max(min_substeps, n_substep[i])
#     for i in range(len(n_substep)):
#         if i % 25 != 0:
#             n_substep[i] = max(n_substep[i], n_substep[i + 1])


@numba.njit()
def method3(remaining_dt, n_cell, cell_start, s):
    end = 0
    for i in range(n_cell - 1, -1, -1):
        if remaining_dt[i] == 0:
            continue
        else:
            end = cell_start[i + 1]
            break
    return end
