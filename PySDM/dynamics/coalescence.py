"""
Collisional coalescence of a superdroplet pair
"""
import numpy as np
from PySDM.dynamics.collision import Collision
from PySDM.physics import si
from PySDM.physics.coalescence_efficiencies import ConstEc
from PySDM.physics.breakup_efficiencies import ConstEb
from PySDM.physics.breakup_fragmentations import AlwaysN
from PySDM.dynamics.impl.random_generator_optimizer import RandomGeneratorOptimizer
import warnings

default_dt_coal_range = (.1 * si.second, 100 * si.second)

class Coalescence(Collision):

    def __init__(self,
                 kernel,
                 coal_eff=ConstEc(Ec=1),
                 seed=None,
                 croupier=None,
                 optimized_random=False,
                 substeps: int = 1,
                 adaptive: bool = False,
                 dt_coal_range=default_dt_coal_range
                 ):
        break_eff = ConstEb(Eb=0)
        fragmentation = AlwaysN(n=1)
        super().__init__(
                 kernel,    # collision kernel
                 coal_eff,  # coalescence efficiency
                 break_eff, # breakup efficiency
                 fragmentation, # fragmentation function
                 seed=seed,
                 croupier=croupier,
                 optimized_random=optimized_random,
                 substeps = substeps,
                 adaptive = adaptive,
                 dt_coal_range=dt_coal_range
                 )
'''

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
        pairs_rand, rand = self.rnd_opt.get_random_arrays()
        self.toss_pairs(self.is_first_in_pair, pairs_rand)
        self.compute_probability(self.prob, self.is_first_in_pair)
        self.compute_gamma(self.prob, rand, self.is_first_in_pair)
        self.core.particles.coalescence(gamma=self.prob, is_first_in_pair=self.is_first_in_pair)
        if self.adaptive:
            self.core.particles.cut_working_length(self.core.particles.adaptive_sdm_end(self.dt_left))

    def toss_pairs(self, is_first_in_pair, u01):
        self.core.particles.permutation(u01, self.croupier == 'local')
        is_first_in_pair.update(
            self.core.particles.cell_start,
            self.core.particles.cell_idx,
            self.core.particles['cell id']
        )
        self.core.particles.sort_within_pair_by_attr(is_first_in_pair, attr_name="n")

    def compute_probability(self, prob, is_first_in_pair):
        self.kernel(self.kernel_temp, is_first_in_pair)
        self.coal_eff(self.coal_eff_temp, is_first_in_pair)
        prob.max(self.core.particles['n'], is_first_in_pair)
        prob *= self.kernel_temp
        prob *= self.coal_eff_temp

        self.core.normalize(prob, self.norm_factor_temp)

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

        self.core.backend.compute_gamma(
            prob,
            rand,
            self.core.particles['n'],
            self.core.particles["cell id"],
            self.collision_rate_deficit,
            self.collision_rate,
            is_first_in_pair
        ) '''
