"""
collisions-related logic including the `PySDM.dynamics.collisions.collision.Collision`
dynamic and coalescence.
Includes collision kernels, ``PySDM.dynamics.collisions.collision_kernels`,
as well as coalescence efficiencies, `PySDM.dynamics.collisions.coalescence_efficiencies`,
and breakup efficiencies `PySDM.dynamics.collisions.breakup_efficiencies`, and
breakup fragmentations `PySDM.dynamics.collisions.breakup_fragmentations`
"""

from PySDM.dynamics.collisions.collision import Breakup, Coalescence, Collision

from . import collision_kernels
