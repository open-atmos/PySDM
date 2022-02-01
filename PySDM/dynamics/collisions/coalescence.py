"""
Collisional coalescence of a superdroplet pair
"""
import warnings
from collections import namedtuple
import numpy as np
from PySDM.dynamics.collisions import Collision
from PySDM.physics import si
from PySDM.physics.coalescence_efficiencies import ConstEc
from PySDM.physics.breakup_efficiencies import ConstEb
from PySDM.physics.breakup_fragmentations import AlwaysN
from PySDM.dynamics.impl.random_generator_optimizer import RandomGeneratorOptimizer

DEFAULTS = namedtuple("_", ('dt_coal_range',))(
  dt_coal_range=(.1 * si.second, 100 * si.second)
)

class Coalescence(Collision):

    def __init__(self,
                 kernel,
                 coal_eff=ConstEc(Ec=1),
                 seed=None,
                 croupier=None,
                 optimized_random=False,
                 substeps: int = 1,
                 adaptive: bool = True,
                 dt_coal_range=DEFAULTS.dt_coal_range
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
