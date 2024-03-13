""" tests calculation of particle Reynolds number """

import pytest
import numpy as np
from PySDM.environments import Box
from PySDM import Builder, Formulae
from PySDM.physics import si


@pytest.mark.parametrize("water_mass", (np.asarray([1 * si.ug, 100 * si.ug]),))
def test_reynolds_number(water_mass, backend_class):
    # arrange
    env = Box(dt=None, dv=None)
    formulae_enabling_reynolds_number_calculation = Formulae(
        ventilation="Froessling1938"
    )
    builder = Builder(
        backend=backend_class(formulae_enabling_reynolds_number_calculation),
        n_sd=water_mass.size,
        environment=env,
    )

    env["air dynamic viscosity"] = 2e-5 * si.Pa * si.s
    env["air density"] = 1 * si.kg / si.m**3

    builder.request_attribute("Reynolds number")
    particulator = builder.build(
        attributes={"water mass": water_mass, "multiplicity": np.ones_like(water_mass)}
    )

    # act
    re_actual = particulator.attributes["Reynolds number"].to_ndarray()

    # assert
    assert (1 < re_actual).all()
    assert (re_actual < 100).all()
