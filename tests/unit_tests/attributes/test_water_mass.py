# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.environments import Box


@pytest.mark.parametrize("mass", (np.asarray([44, 666]),))
def test_water_mass(mass, backend_class_with_jax):
    # arrange
    backend = backend_class_with_jax(
        Formulae(particle_shape_and_density="MixedPhaseSpheres")
    )
    env = Box(dt=None, dv=None)
    builder = Builder(backend=backend, n_sd=mass.size, environment=env)
    builder.request_attribute("water mass")
    particulator = builder.build(
        attributes={"signed water mass": -mass, "multiplicity": np.ones_like(mass)}
    )

    # act
    mass_actual = particulator.attributes["water mass"].to_ndarray()

    # assert
    np.testing.assert_allclose(mass_actual, mass)
