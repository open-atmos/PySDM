"""
Created at 05.13.21 by edejong
"""

import numpy as np
from PySDM.physics import si
from PySDM.dynamics.impl.random_generator_optimizer import RandomGeneratorOptimizer
import warnings

default_dt_coal_range = (.1 * si.second, 100 * si.second)


class Breakup:

    def __init__(self,
                 kernel,
                 fragmentation,
                 croupier=None,
                 optimized_random=False,
                 substeps: int = 1,
                 adaptive: bool = False,
                 dt_coal_range=default_dt_coal_range
                 ):
        assert substeps == 1 or adaptive is False

        self.core = None
        self.enable = True

        self.kernel = kernel
        self.fragmentation = fragmentation

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
        self.core = builder.core
        self.rnd_opt = RandomGeneratorOptimizer(optimized_random=self.optimized_random,
                                                dt_min=self.dt_coal_range[0],
                                                seed=self.core.formulae.seed)
        self.optimised_random = None

        if self.core.n_sd < 2:
            raise ValueError("No one to collide with!")
        if self.dt_coal_range[1] > self.core.dt:
            self.dt_coal_range = (self.dt_coal_range[0], self.core.dt)
        assert self.dt_coal_range[0] <= self.dt_coal_range[1]

        self.kernel_temp = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.norm_factor_temp = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)  # TODO #372
        self.prob = self.core.PairwiseStorage.empty(self.core.n_sd // 2, dtype=float)
        self.is_first_in_pair = self.core.PairIndicator(self.core.n_sd)
        self.dt_left = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)

        self.stats_n_substep = self.core.Storage.empty(self.core.mesh.n_cell, dtype=int)
        self.stats_n_substep[:] = 0 if self.adaptive else self.__substeps
        self.stats_dt_min = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)
        self.stats_dt_min[:] = np.nan

        self.rnd_opt.register(builder)
        self.kernel.register(builder)

        if self.croupier is None:
            self.croupier = self.core.backend.default_croupier
        
        
        # TODO Emily: check whether this is still correct to use.
        self.collision_rate = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))
        self.collision_rate_deficit = self.core.Storage.from_ndarray(np.zeros(self.core.mesh.n_cell, dtype=int))

    def __call__(self):
        if self.enable:
            if not self.adaptive:
                for _ in range(self.__substeps):
                    self.step()
            else:
                self.dt_left[:] = self.core.dt

                while self.core.particles.get_working_length() != 0:
                    self.core.particles.cell_idx.sort_by_key(self.dt_left)
                    self.step()

                self.core.particles.reset_working_length()
                self.core.particles.reset_cell_idx()
            self.rnd_opt.reset()

    def step(self):
        # (1) Make the superdroplet list 
        pairs_rand, rand = self.rnd_opt.get_random_arrays()
        # (2) candidate-pair list
        self.toss_pairs(self.is_first_in_pair, pairs_rand)
        # (3) Compute the probability of a collision
        self.compute_probability(self.prob, self.is_first_in_pair)
        # (4) Compute gamma...
        self.compute_gamma(self.prob, rand, self.is_first_in_pair)
        # (5) Perform the coalescence step: 
        # ../../state/particles.py
        # ../backends/*/_algorithmic_methods.py
        self.core.particles.coalescence(gamma=self.prob, is_first_in_pair=self.is_first_in_pair)
        # TODO Emily: 
        # (7) self.core.particles.n_fragment()
        # (6) self.core.particles.breakup()
        
        if self.adaptive:
            self.core.particles.cut_working_length(self.core.particles.adaptive_sdm_end(self.dt_left))

    # (2) candidate-pair list: put them in order by multiplicity
    def toss_pairs(self, is_first_in_pair, u01):
        self.core.particles.permutation(u01, self.croupier == 'local')
        is_first_in_pair.update(
            self.core.particles.cell_start,
            self.core.particles.cell_idx,
            self.core.particles['cell id']
        )
        self.core.particles.sort_within_pair_by_attr(is_first_in_pair, attr_name="n")

    # (3) Compute probability of a collision
    def compute_probability(self, prob, is_first_in_pair):
        self.kernel(self.kernel_temp, is_first_in_pair)
        # P_jk = max(xi_j, xi_k)*P_jk
        prob.max(self.core.particles['n'], is_first_in_pair)
        prob *= self.kernel_temp

        self.core.normalize(prob, self.norm_factor_temp)

    # (4) Compute gamma, i.e. whether the collision succeeds
    def compute_gamma(self, prob, rand, is_first_in_pair):
        if self.adaptive:
            self.core.backend.adaptive_sdm_gamma(
                prob,
                self.core.particles['n'],
                self.core.particles["cell id"],
                self.dt_left,
                self.core.dt,
                self.dt_coal_range,
                is_first_in_pair,
                self.stats_n_substep,
                self.stats_dt_min
            )
            if self.stats_dt_min.amin() == self.dt_coal_range[0]:
                warnings.warn("coalescence adaptive time-step reached dt_min")
        else:
            prob /= self.__substeps
            
        # src is ../backends/numba/impl/_algorithmic_methods.py, line 149
        self.core.backend.compute_gamma(
            prob,
            rand,
            self.core.particles['n'],
            self.core.particles["cell id"],
            self.collision_rate_deficit,
            self.collision_rate,
            is_first_in_pair
        )
