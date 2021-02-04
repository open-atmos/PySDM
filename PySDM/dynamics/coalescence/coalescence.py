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
                 substeps: int = 1,  # TODO: meaning for adaptive=True?
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
        self.dt_left = None

        self.actual_length = None

    def register(self, builder):
        self.core = builder.core

        if self.core.n_sd < 2:
            raise ValueError("No one to collide with!")
        # TODO
        # if self.adaptive:
        #     assert self.core.dt >= self.dt_coal_range[0]
        #     assert self.core.dt >= self.dt_coal_range[1]

        self.kernel_temp = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.norm_factor_temp = self.core.Storage.empty(self.core.n_sd, dtype=float)  # TODO #372
        self.prob = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.is_first_in_pair = self.core.PairIndicator(self.core.n_sd)
        self.dt_left = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)

        self.n_substep = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.n_substep[:] = self.__substeps

        self.rnd_opt.register(builder)
        self.kernel.register(builder)

        if self.croupier is None:
            self.croupier = self.core.backend.default_croupier

        self.collision_rate = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.collision_rate_deficit = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

    def __call__(self):
        if self.enable:
            if not self.adaptive:
                for s in range(self.__substeps):  # TODO
                    self.step(s)
            else:
                self.dt_left[:] = self.core.dt
                self.actual_length = self.core.particles._Particles__idx.length
                self.actual_cell_idx = self.core.particles.cell_idx.data

                s = 0
                while self.core.particles._Particles__idx.length != 0:
                    self.core.particles.cell_idx.data = self.dt_left.data.argsort(kind="stable")[::-1]
                    self.step(s)
                    s += 1

                self.core.particles._Particles__idx.length = self.actual_length
                self.core.particles.cell_idx.data = self.actual_cell_idx
                self.core.particles._Particles__sort_by_cell_id()

    def step(self, s):
        pairs_rand, rand = self.rnd_opt.get_random_arrays(s)
        self.toss_pairs(self.is_first_in_pair, pairs_rand, s)
        self.compute_probability(self.prob, self.is_first_in_pair)
        self.compute_gamma(self.prob, rand, self.is_first_in_pair)
        self.core.particles.coalescence(gamma=self.prob, is_first_in_pair=self.is_first_in_pair)

    def toss_pairs(self, is_first_in_pair, u01, s):
        if self.adaptive:
            self.core.particles._Particles__idx.length = self.actual_length
        self.core.particles.sanitize()
        if self.adaptive:
            self.actual_length = self.core.particles._Particles__idx.length

        self.core.particles.permutation(u01, self.croupier == 'local')

        if self.adaptive:
            # TODO: compute it earlier
            end = self.core.particles.adaptive_sdm_end(self.dt_left)
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

    def compute_gamma(self, prob, rand, is_first_in_pair):
        if self.adaptive:
            # TODO: take into account max_dt and min_dt range
            self.core.backend.adaptive_sdm_gamma(prob, self.core.particles._Particles__idx, self.core.particles['n'],
                                        self.core.particles["cell id"],
                                        self.dt_left, self.core.dt, is_first_in_pair)

        self.core.backend.compute_gamma(prob, rand, self.core.particles._Particles__idx, self.core.particles['n'],
                                        self.core.particles["cell id"],
                                        self.collision_rate_deficit, self.collision_rate, is_first_in_pair)



