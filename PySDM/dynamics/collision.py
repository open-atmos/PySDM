"""
Created at 09.30.21 by edejong
"""

import numpy as np
from PySDM.physics import si
from PySDM.dynamics.impl.random_generator_optimizer import RandomGeneratorOptimizer
from PySDM.dynamics.impl.random_generator_optimizer_nopair import RandomGeneratorOptimizerNoPair
import warnings

default_dt_coal_range = (.1 * si.second, 100 * si.second)

"""
General algorithm format:
1. Determine whether collision occurs
2. If collision occurs:
    a. Determine whether coalescence, breakup, or bouncing occur
        Ec = coalescence efficiency
        Eb = collisional-breakup efficiency
        1 - Ec - Eb = bounce back to original fragments (subset of breakup)
    b. Perform the relevant dynamic
"""

class Collision:

    def __init__(self,
                 kernel,    # collision kernel
                 coal_eff,  # coalescence efficiency
                 break_eff, # breakup efficiency
                 fragmentation, # fragmentation function
                 seed=None,
                 croupier=None,
                 optimized_random=False,
                 substeps: int = 1,
                 adaptive: bool = False,
                 dt_coal_range=default_dt_coal_range
                 ):
        assert substeps == 1 or adaptive is False

        self.particulator = None
        self.enable = True

        self.kernel = kernel
        self.coal_eff = coal_eff
        self.break_eff = break_eff
        self.fragmentation = fragmentation
        self.seed = seed

        assert dt_coal_range[0] > 0
        self.croupier = croupier
        self.optimized_random = optimized_random
        self.__substeps = substeps
        self.adaptive = adaptive
        self.stats_n_substep = None
        self.stats_dt_min = None
        self.dt_coal_range = tuple(dt_coal_range)

        self.kernel_temp = None         # stores the output from calling self.kernel()
        self.n_fragment = None 
        self.Ec_temp = None
        self.Eb_temp = None
        self.dyn = None
        self.neg_ones = None
        self.norm_factor_temp = None
        self.prob = None
        self.is_first_in_pair = None
        self.dt_left = None

        self.collision_rate = None
        self.collision_rate_deficit = None
        self.coalescence_rate = None
        self.breakup_rate = None
                

    def register(self, builder):
        self.particulator = builder.particulator
        # determine whether collision occurs
        self.rnd_opt_coll = RandomGeneratorOptimizer(optimized_random=self.optimized_random,
                                                dt_min=self.dt_coal_range[0],
                                                seed=self.seed) #self.particulator.formulae.seed+1)
        # determine which process occurs
        self.rnd_opt_proc = RandomGeneratorOptimizerNoPair(optimized_random=self.optimized_random,
                                                dt_min=self.dt_coal_range[0],
                                                seed=self.seed) #self.particulator.formulae.seed+1)
        # for generating number of fragments
        self.rnd_opt_frag = RandomGeneratorOptimizerNoPair(optimized_random=self.optimized_random,
                                                dt_min=self.dt_coal_range[0],
                                                seed=self.seed) #self.particulator.formulae.seed+1)
        self.optimised_random = None

        if self.particulator.n_sd < 2:
            raise ValueError("No one to collide with!")
        if self.dt_coal_range[1] > self.particulator.dt:
            self.dt_coal_range = (self.dt_coal_range[0], self.particulator.dt)
        assert self.dt_coal_range[0] <= self.dt_coal_range[1]

        self.kernel_temp = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.n_fragment = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.Ec_temp = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.Eb_temp = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.dyn = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        neg_ones_tmp = np.tile([-1], self.particulator.n_sd // 2)
        self.neg_ones = self.particulator.PairwiseStorage.from_ndarray(neg_ones_tmp)
        self.norm_factor_temp = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=float)
        self.prob = self.particulator.PairwiseStorage.empty(self.particulator.n_sd // 2, dtype=float)
        self.is_first_in_pair = self.particulator.PairIndicator(self.particulator.n_sd)
        self.dt_left = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=float)

        self.stats_n_substep = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=int)
        self.stats_n_substep[:] = 0 if self.adaptive else self.__substeps
        self.stats_dt_min = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=float)
        self.stats_dt_min[:] = np.nan

        self.rnd_opt_coll.register(builder)
        self.rnd_opt_proc.register(builder)
        self.rnd_opt_frag.register(builder)
        self.kernel.register(builder)
        self.coal_eff.register(builder)
        self.break_eff.register(builder)
        self.fragmentation.register(builder)

        if self.croupier is None:
            self.croupier = self.particulator.backend.default_croupier
        
        self.collision_rate = self.particulator.Storage.from_ndarray(np.zeros(self.particulator.mesh.n_cell, dtype=int))
        self.collision_rate_deficit = self.particulator.Storage.from_ndarray(np.zeros(self.particulator.mesh.n_cell, dtype=int))
        self.coalescence_rate = self.particulator.Storage.from_ndarray(np.zeros(self.particulator.mesh.n_cell, dtype=int))
        self.breakup_rate = self.particulator.Storage.from_ndarray(np.zeros(self.particulator.mesh.n_cell, dtype=int))

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
            self.rnd_opt_coll.reset()
            self.rnd_opt_proc.reset()
            self.rnd_opt_frag.reset()

    def step(self):
        # (1) Make the superdroplet list and random numbers for collision, process, and fragmentation
        pairs_rand, rand = self.rnd_opt_coll.get_random_arrays()
        proc_rand = self.rnd_opt_proc.get_random_arrays()
        rand_frag = self.rnd_opt_frag.get_random_arrays()
        
        # (2) candidate-pair list
        self.toss_pairs(self.is_first_in_pair, pairs_rand)
        
        # (3a) Compute the probability of a collision
        self.compute_probability(self.prob, self.is_first_in_pair)
        
        # (3b) Compute the coalescence and breakup efficiences
        self.coal_eff(self.Ec_temp, self.is_first_in_pair)
        self.break_eff(self.Eb_temp, self.is_first_in_pair)
        
        # (3c) Compute the number of fragments
        self.fragmentation(self.n_fragment, rand_frag, self.is_first_in_pair)
        
        # (4) Compute gamma...
        self.compute_gamma(self.prob, rand, self.is_first_in_pair)
        
        # (5) Perform the collisional-coalescence/breakup step: 
        self.particulator.collision(gamma=self.prob, rand=proc_rand, dyn=self.dyn, Ec=self.Ec_temp, Eb=self.Eb_temp, n_fragment=self.n_fragment, 
                                    cell_id=self.particulator.attributes["cell id"], coalescence_rate=self.coalescence_rate, 
                                    breakup_rate=self.breakup_rate, is_first_in_pair=self.is_first_in_pair)
        
        if self.adaptive:
            self.particulator.attributes.cut_working_length(self.particulator.attributes.adaptive_sdm_end(self.dt_left))

    # (2) candidate-pair list: put them in order by multiplicity
    def toss_pairs(self, is_first_in_pair, u01):
        self.particulator.attributes.permutation(u01, self.croupier == 'local')
        is_first_in_pair.update(
            self.particulator.attributes.cell_start,
            self.particulator.attributes.cell_idx,
            self.particulator.attributes['cell id']
        )
        self.particulator.sort_within_pair_by_attr(is_first_in_pair, attr_name="n")

    # (3a) Compute probability of a collision
    def compute_probability(self, prob, is_first_in_pair):
        self.kernel(self.kernel_temp, is_first_in_pair)
        # P_jk = max(xi_j, xi_k)*P_jk*E_c
        prob.max(self.particulator.attributes['n'], is_first_in_pair)
        prob *= self.kernel_temp

        self.particulator.normalize(prob, self.norm_factor_temp)
        
    # (3c) Compute n_fragment
    def compute_n_fragment(self, n_fragment, u01, is_first_in_pair):
        self.fragmentation(n_fragment, u01, is_first_in_pair)

    # (4) Compute gamma, i.e. whether the collision leads to breakup
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
                warnings.warn("breakup adaptive time-step reached dt_min")
        else:
            prob /= self.__substeps
            
        # src is ../backends/numba/impl/_algorithmic_methods.py, line 149
        self.particulator.backend.compute_gamma(
            prob,
            rand,
            self.particulator.attributes['n'],
            self.particulator.attributes["cell id"],
            self.collision_rate_deficit,
            self.collision_rate,
            is_first_in_pair
        )
        

