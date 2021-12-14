"""
Collisional breakup of a superdroplet pair
Created at 05.13.21 by edejong
"""

import numpy as np
from PySDM.dynamics.collision import Collision
from PySDM.physics import si
from PySDM.dynamics.impl.random_generator_optimizer import RandomGeneratorOptimizer
from PySDM.dynamics.impl.random_generator_optimizer_nopair import RandomGeneratorOptimizerNoPair
from PySDM.physics.coalescence_efficiencies import ConstEc
from PySDM.physics.breakup_efficiencies import ConstEb
import warnings

default_dt_coal_range = (.1 * si.second, 100 * si.second)


class Breakup(Collision):

    def __init__(self,
                 kernel,
                 fragmentation,
                 seed=None,
                 croupier=None,
                 optimized_random=False,
                 substeps: int = 1,
                 adaptive: bool = False,
                 dt_coal_range=default_dt_coal_range
                 ):
        coal_eff = ConstEc(Ec = 0.0)
        break_eff = ConstEb(Eb = 1.0)
        super().__init__(
                kernel,
                coal_eff,
                break_eff,
                fragmentation,
                seed=seed,
                croupier=croupier,
                optimized_random=optimized_random,
                substeps=substeps,
                adaptive=adaptive,
                dt_coal_range=dt_coal_range
    )