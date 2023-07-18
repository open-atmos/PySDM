import pytest
import numpy as np
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.builder import Builder
from PySDM.dynamics.collisions.collision import Coalescence
from PySDM.dynamics.collisions.collision_kernels.constantK import ConstantK
from PySDM.physics import si
from PySDM.environments.box import Box

def test_small_timescale(default_attributes, backend):
    builder = Builder(n_sd=len(default_attributes["n"]), backend=backend_class())
    builder.set_environment(Box(dt=1, dv=1))
    builder.request_attribute("fall velocity")
    particulator = builder.build(
        attributes=default_attributes,
        products=()
    )

def test_large_timescale(default_attributes, backend):
    pass

def test_end_behavior(default_attributes, backend):
    pass

def test_exponential_decay(default_attributes, backend):
    pass
