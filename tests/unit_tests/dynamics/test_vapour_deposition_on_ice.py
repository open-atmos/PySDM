import numpy as np

import pytest

from PySDM.physics import si
from PySDM.backends import CPU
from PySDM import Builder
from PySDM import Formulae
from PySDM.environments import Box
from PySDM.dynamics import VapourDepositionOnIce
from PySDM.products import IceWaterContent


@pytest.mark.parametrize('dt', (1 * si.s, .1 * si.s))
@pytest.mark.parametrize('water_mass', (-si.ng, -si.ug, -si.mg, si.mg))
@pytest.mark.parametrize('fastmath', (True, False))
def test_TODO(dt, water_mass, fastmath, dv=1*si.m**3):
    # arrange
    n_sd = 1
    builder = Builder(
        n_sd=n_sd,
        environment=Box(dt=dt, dv=dv),
        backend=CPU(formulae=Formulae(
            fastmath=fastmath,
            particle_shape_and_density="MixedPhaseSpheres"
        ))
    )
    deposition = VapourDepositionOnIce()
    builder.add_dynamic(deposition)
    particulator = builder.build(
        attributes={
            'multiplicity': np.full(shape=(n_sd,), fill_value=1),
            'water mass': np.full(shape=(n_sd,), fill_value=water_mass),
        },
        products=(IceWaterContent(),)
    )

    # act
    iwc_old = particulator.products['ice water content'].get().copy()
    particulator.run(steps=1)
    iwc_new = particulator.products['ice water content'].get().copy()

    # assert
    assert (iwc_new > iwc_old).all()