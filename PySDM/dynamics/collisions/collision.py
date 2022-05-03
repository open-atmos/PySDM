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
import warnings
from collections import namedtuple

import numpy as np

from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.impl.random_generator_optimizer import RandomGeneratorOptimizer
from PySDM.dynamics.impl.random_generator_optimizer_nopair import (
    RandomGeneratorOptimizerNoPair,
)
from PySDM.physics import si

DEFAULTS = namedtuple("_", ("dt_coal_range",))(
    dt_coal_range=(0.1 * si.second, 100.0 * si.second)
)


class Collision:
    def __init__(
        self,
        *,
        collision_kernel,
        coalescence_efficiency,
        breakup_efficiency,
        fragmentation_function,
        croupier=None,
        optimized_random=False,
        substeps: int = 1,
        adaptive: bool = True,
        dt_coal_range=DEFAULTS.dt_coal_range,
        enable_breakup: bool = True,
    ):
        assert substeps == 1 or adaptive is False

        self.particulator = None

        self.enable = True
        self.enable_breakup = enable_breakup

        self.collision_kernel = collision_kernel
        self.compute_coalescence_efficiency = coalescence_efficiency
        self.compute_breakup_efficiency = breakup_efficiency
        self.compute_number_of_fragments = fragmentation_function

        self.rnd_opt_frag = None
        self.rnd_opt_coll = None
        self.rnd_opt_proc = None
        self.optimised_random = None

        assert dt_coal_range[0] > 0
        self.croupier = croupier
        self.optimized_random = optimized_random
        self.__substeps = substeps
        self.adaptive = adaptive
        self.stats_n_substep = None
        self.stats_dt_min = None
        self.dt_coal_range = tuple(dt_coal_range)

        self.kernel_temp = None
        self.n_fragment = None
        self.Ec_temp = None
        self.Eb_temp = None
        self.norm_factor_temp = None
        self.prob = None
        self.is_first_in_pair = None
        self.dt_left = None

        self.collision_rate = None
        self.collision_rate_deficit = None
        self.coalescence_rate = None
        self.breakup_rate = None
        self.breakup_rate_deficit = None

    def register(self, builder):
        self.particulator = builder.particulator
        rnd_args = {
            "optimized_random": self.optimized_random,
            "dt_min": self.dt_coal_range[0],
            "seed": builder.formulae.seed,
        }
        self.rnd_opt_coll = RandomGeneratorOptimizer(**rnd_args)
        if self.enable_breakup:
            self.rnd_opt_proc = RandomGeneratorOptimizerNoPair(**rnd_args)
            self.rnd_opt_frag = RandomGeneratorOptimizerNoPair(**rnd_args)

        if self.particulator.n_sd < 2:
            raise ValueError("No one to collide with!")
        if self.dt_coal_range[1] > self.particulator.dt:
            self.dt_coal_range = (self.dt_coal_range[0], self.particulator.dt)
        assert self.dt_coal_range[0] <= self.dt_coal_range[1]

        empty_args_pairwise = {"shape": self.particulator.n_sd // 2, "dtype": float}
        empty_args_cellwise = {"shape": self.particulator.mesh.n_cell, "dtype": float}
        self.kernel_temp = self.particulator.PairwiseStorage.empty(
            **empty_args_pairwise
        )
        self.norm_factor_temp = self.particulator.Storage.empty(**empty_args_cellwise)
        self.prob = self.particulator.PairwiseStorage.empty(**empty_args_pairwise)
        self.is_first_in_pair = self.particulator.PairIndicator(self.particulator.n_sd)
        self.dt_left = self.particulator.Storage.empty(**empty_args_cellwise)

        self.stats_n_substep = self.particulator.Storage.empty(
            self.particulator.mesh.n_cell, dtype=int
        )
        self.stats_n_substep[:] = 0 if self.adaptive else self.__substeps
        self.stats_dt_min = self.particulator.Storage.empty(**empty_args_cellwise)
        self.stats_dt_min[:] = np.nan

        self.rnd_opt_coll.register(builder)
        self.collision_kernel.register(builder)

        if self.croupier is None:
            self.croupier = self.particulator.backend.default_croupier

        counter_args = (np.zeros(self.particulator.mesh.n_cell, dtype=int),)
        self.collision_rate = self.particulator.Storage.from_ndarray(*counter_args)
        self.collision_rate_deficit = self.particulator.Storage.from_ndarray(
            *counter_args
        )
        self.coalescence_rate = self.particulator.Storage.from_ndarray(*counter_args)

        if self.enable_breakup:
            self.n_fragment = self.particulator.PairwiseStorage.empty(
                **empty_args_pairwise
            )
            self.Ec_temp = self.particulator.PairwiseStorage.empty(
                **empty_args_pairwise
            )
            self.Eb_temp = self.particulator.PairwiseStorage.empty(
                **empty_args_pairwise
            )
            self.rnd_opt_proc.register(builder)
            self.rnd_opt_frag.register(builder)
            self.compute_coalescence_efficiency.register(builder)
            self.compute_breakup_efficiency.register(builder)
            self.compute_number_of_fragments.register(builder)
            self.breakup_rate = self.particulator.Storage.from_ndarray(*counter_args)
            self.breakup_rate_deficit = self.particulator.Storage.from_ndarray(
                *counter_args
            )

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
            if self.enable_breakup:
                self.rnd_opt_proc.reset()
                self.rnd_opt_frag.reset()

    def step(self):
        pairs_rand, rand = self.rnd_opt_coll.get_random_arrays()

        self.toss_candidate_pairs_and_sort_within_pair_by_multiplicity(
            self.is_first_in_pair, pairs_rand
        )

        self.compute_probabilities_of_collision(self.prob, self.is_first_in_pair)
        if self.enable_breakup:
            proc_rand = self.rnd_opt_proc.get_random_arrays()
            rand_frag = self.rnd_opt_frag.get_random_arrays()
            self.compute_coalescence_efficiency(self.Ec_temp, self.is_first_in_pair)
            self.compute_breakup_efficiency(self.Eb_temp, self.is_first_in_pair)
            self.compute_number_of_fragments(
                self.n_fragment, rand_frag, self.is_first_in_pair
            )
        else:
            proc_rand = None

        self.compute_gamma(self.prob, rand, self.is_first_in_pair)

        self.particulator.collision_coalescence_breakup(
            enable_breakup=self.enable_breakup,
            gamma=self.prob,
            rand=proc_rand,
            Ec=self.Ec_temp,
            Eb=self.Eb_temp,
            n_fragment=self.n_fragment,
            coalescence_rate=self.coalescence_rate,
            breakup_rate=self.breakup_rate,
            breakup_rate_deficit=self.breakup_rate_deficit,
            is_first_in_pair=self.is_first_in_pair,
        )

        if self.adaptive:
            self.particulator.attributes.cut_working_length(
                self.particulator.adaptive_sdm_end(self.dt_left)
            )

    def toss_candidate_pairs_and_sort_within_pair_by_multiplicity(
        self, is_first_in_pair, u01
    ):
        self.particulator.attributes.permutation(u01, self.croupier == "local")
        is_first_in_pair.update(
            self.particulator.attributes.cell_start,
            self.particulator.attributes.cell_idx,
            self.particulator.attributes["cell id"],
        )
        self.particulator.sort_within_pair_by_attr(is_first_in_pair, attr_name="n")

    def compute_probabilities_of_collision(self, prob, is_first_in_pair):
        """eq. (20) in [Shima et al. 2009](https://doi.org/10.1002/qj.441)"""
        self.collision_kernel(self.kernel_temp, is_first_in_pair)
        prob.max(self.particulator.attributes["n"], is_first_in_pair)
        prob *= self.kernel_temp

        self.particulator.normalize(prob, self.norm_factor_temp)

    def compute_n_fragment(self, n_fragment, u01, is_first_in_pair):
        self.compute_number_of_fragments(n_fragment, u01, is_first_in_pair)

    def compute_gamma(self, prob, rand, is_first_in_pair):
        """see sec. 5.1.3 point (3) in [Shima et al. 2009](https://doi.org/10.1002/qj.441)
        note that in PySDM gamma also serves the purpose of disabling collisions
        for droplets without a pair (i.e. odd number of particles within a grid cell)
        """
        if self.adaptive:
            self.particulator.backend.adaptive_sdm_gamma(
                gamma=prob,
                n=self.particulator.attributes["n"],
                cell_id=self.particulator.attributes["cell id"],
                dt_left=self.dt_left,
                dt=self.particulator.dt,
                dt_range=self.dt_coal_range,
                is_first_in_pair=is_first_in_pair,
                stats_n_substep=self.stats_n_substep,
                stats_dt_min=self.stats_dt_min,
            )
            if self.stats_dt_min.amin() == self.dt_coal_range[0]:
                warnings.warn("adaptive time-step reached dt_min")
        else:
            prob /= self.__substeps

        self.particulator.backend.compute_gamma(
            gamma=prob,
            rand=rand,
            multiplicity=self.particulator.attributes["n"],
            cell_id=self.particulator.attributes["cell id"],
            collision_rate_deficit=self.collision_rate_deficit,
            collision_rate=self.collision_rate,
            is_first_in_pair=is_first_in_pair,
        )


class Coalescence(Collision):
    def __init__(
        self,
        *,
        collision_kernel,
        coalescence_efficiency=ConstEc(Ec=1),
        croupier=None,
        optimized_random=False,
        substeps: int = 1,
        adaptive: bool = True,
        dt_coal_range=DEFAULTS.dt_coal_range,
    ):
        breakup_efficiency = ConstEb(Eb=0)
        fragmentation_function = AlwaysN(n=1)
        super().__init__(
            collision_kernel=collision_kernel,
            coalescence_efficiency=coalescence_efficiency,
            breakup_efficiency=breakup_efficiency,
            fragmentation_function=fragmentation_function,
            croupier=croupier,
            optimized_random=optimized_random,
            substeps=substeps,
            adaptive=adaptive,
            dt_coal_range=dt_coal_range,
            enable_breakup=False,
        )


class Breakup(Collision):
    def __init__(
        self,
        *,
        collision_kernel,
        fragmentation_function,
        croupier=None,
        optimized_random=False,
        substeps: int = 1,
        adaptive: bool = True,
        dt_coal_range=DEFAULTS.dt_coal_range,
    ):
        coalescence_efficiency = ConstEc(Ec=0.0)
        breakup_efficiency = ConstEb(Eb=1.0)
        super().__init__(
            collision_kernel=collision_kernel,
            coalescence_efficiency=coalescence_efficiency,
            breakup_efficiency=breakup_efficiency,
            fragmentation_function=fragmentation_function,
            croupier=croupier,
            optimized_random=optimized_random,
            substeps=substeps,
            adaptive=adaptive,
            dt_coal_range=dt_coal_range,
        )
