"""
SDM implementation with adaptive timestepping
"""
import numpy as np
from PySDM.physics import si
from PySDM.dynamics.impl.random_generator_optimizer import RandomGeneratorOptimizer
import warnings

default_dt_coal_range = (.1 * si.second, 100 * si.second)


class Coalescence:

    def __init__(self,
                 kernel,
                 croupier=None,
                 optimized_random=False,
                 substeps: int = 1,
                 adaptive: bool = True,
                 dt_coal_range=default_dt_coal_range
                 ):
        assert substeps == 1 or adaptive is False

        self.particulator = None
        self.enable = True

        self.kernel = kernel

        assert dt_coal_range[0] > 0
        self.croupier = croupier
        self.optimized_random = optimized_random
        self.__substeps = substeps
        self.adaptive = adaptive
        self.stats_n_substep = None
        self.stats_dt_min = None
        self.dt_coal_range = tuple(dt_coal_range)

        self.kernel_temp = None
        self.norm_factor_temp = None
        self.prob = None
        self.is_first_in_pair = None
        self.dt_left = None

        self.collision_rate = None
        self.collision_rate_deficit = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.rnd_opt = RandomGeneratorOptimizer(optimized_random=self.optimized_random,
                                                dt_min=self.dt_coal_range[0],
                                                seed=self.particulator.formulae.seed)
        self.optimised_random = None

        if self.particulator.n_sd < 2:
            raise ValueError("No one to collide with!")
        if self.dt_coal_range[1] > self.particulator.dt:
            self.dt_coal_range = (self.dt_coal_range[0], self.particulator.dt)
        assert self.dt_coal_range[0] <= self.dt_coal_range[1]

        self.kernel_temp = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.norm_factor_temp = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=float)
        self.prob = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.is_first_in_pair = self.particulator.PairIndicator(self.particulator.n_sd)
        self.dt_left = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=float)

        self.stats_n_substep = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=int)
        self.stats_n_substep[:] = 0 if self.adaptive else self.__substeps
        self.stats_dt_min = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=float)
        self.stats_dt_min[:] = np.nan

        self.rnd_opt.register(builder)
        self.kernel.register(builder)

        if self.croupier is None:
            self.croupier = self.particulator.backend.default_croupier

        self.collision_rate = self.particulator.Storage.from_ndarray(np.zeros(self.particulator.mesh.n_cell, dtype=int))
        self.collision_rate_deficit = self.particulator.Storage.from_ndarray(np.zeros(self.particulator.mesh.n_cell, dtype=int))

    def __call__(self):
        if self.enable:
            if not self.adaptive:
                for _ in range(self.__substeps):
                    self.step()
            else:
                self.dt_left[:] = self.particulator.dt

                while self.particulator.attributes.get_working_length() != 0:
                    self.particulator.attributes.cell_idx.sort_by_key(self.dt_left)
                    self.step()

                self.particulator.attributes.reset_working_length()
                self.particulator.attributes.reset_cell_idx()
            self.rnd_opt.reset()

    def step(self):
        pairs_rand, rand = self.rnd_opt.get_random_arrays()
        self.toss_pairs(self.is_first_in_pair, pairs_rand)
        self.compute_probability(self.prob, self.is_first_in_pair)
        self.compute_gamma(self.prob, rand, self.is_first_in_pair)
        self.particulator.attributes.coalescence(gamma=self.prob, is_first_in_pair=self.is_first_in_pair)
        if self.adaptive:
            self.particulator.attributes.cut_working_length(self.particulator.attributes.adaptive_sdm_end(self.dt_left))

    def toss_pairs(self, is_first_in_pair, u01):
        self.particulator.attributes.permutation(u01, self.croupier == 'local')
        is_first_in_pair.update(
            self.particulator.attributes.cell_start,
            self.particulator.attributes.cell_idx,
            self.particulator.attributes['cell id']
        )
        self.particulator.attributes.sort_within_pair_by_attr(is_first_in_pair, attr_name="n")

    def compute_probability(self, prob, is_first_in_pair):
        self.kernel(self.kernel_temp, is_first_in_pair)
        prob.max(self.particulator.attributes['n'], is_first_in_pair)
        prob *= self.kernel_temp

        self.particulator.normalize(prob, self.norm_factor_temp)

    def compute_gamma(self, prob, rand, is_first_in_pair):
        if self.adaptive:
            self.particulator.backend.adaptive_sdm_gamma(
                prob,
                self.particulator.attributes['n'],
                self.particulator.attributes["cell id"],
                self.dt_left,
                self.particulator.dt,
                self.dt_coal_range,
                is_first_in_pair,
                self.stats_n_substep,
                self.stats_dt_min
            )
            if self.stats_dt_min.amin() == self.dt_coal_range[0]:
                warnings.warn("coalescence adaptive time-step reached dt_min")
        else:
            prob /= self.__substeps

        self.particulator.backend.compute_gamma(
            prob,
            rand,
            self.particulator.attributes['n'],
            self.particulator.attributes["cell id"],
            self.collision_rate_deficit,
            self.collision_rate,
            is_first_in_pair
        )
