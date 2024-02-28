""" tests calculation of particle Reynolds number """

import pytest
import numpy as np
from PySDM.environments import Box
from PySDM import Builder
from PySDM.physics import si
from PySDM.backends import GPU


@pytest.mark.parametrize("water_mass", (np.asarray([1 * si.ug, 100 * si.ug]),))
def test_reynolds_number(water_mass, backend_class):
    if backend_class == GPU:
        pytest.skip("TODO #1282")

    # arrange
    env = Box(dt=None, dv=None)
    builder = Builder(backend=backend_class(), n_sd=water_mass.size, environment=env)

    env["T"] = 300 * si.K
    env["rhod"] = 1 * si.kg / si.m**3
    env["water vapour mixing ratio"] = 1 * si.g / si.kg

    builder.request_attribute("Reynolds number")
    particulator = builder.build(
        attributes={"water mass": water_mass, "multiplicity": np.ones_like(water_mass)}
    )

    # act
    re_actual = particulator.attributes["Reynolds number"].to_ndarray()

    # assert
    assert (1 < re_actual).all()
    assert (re_actual < 100).all()
