# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import struct

import numpy as np
import pytest
from PySDM_examples.Berry_1967.settings import Settings

from PySDM.backends import ThrustRTC
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.dynamics.collisions.collision_kernels import (
    Electric,
    Geometric,
    Golovin,
    Hydrodynamic,
)
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity


@pytest.mark.parametrize("croupier", ("local", "global"))
@pytest.mark.parametrize("adaptive", (True, False))
@pytest.mark.parametrize("kernel", (Geometric(), Electric(), Hydrodynamic()))
def test_coalescence(backend_class, kernel, croupier, adaptive):
    if backend_class == ThrustRTC and croupier == "local":
        pytest.skip("TODO #358")
    if backend_class == ThrustRTC and adaptive and croupier == "global":
        pytest.skip("TODO #329")
    # Arrange
    s = Settings()
    s.formulae.seed = 0
    steps = [0, 800]

    env = Box(dt=s.dt, dv=s.dv)
    builder = Builder(
        n_sd=s.n_sd, backend=backend_class(formulae=s.formulae), environment=env
    )
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
        s.spectrum
    ).sample(s.n_sd)
    builder.add_dynamic(
        Coalescence(collision_kernel=kernel, croupier=croupier, adaptive=adaptive)
    )
    particulator = builder.build(attributes)

    volumes = {}

    # Act
    for step in steps:
        particulator.run(step - particulator.n_steps)
        volumes[particulator.n_steps] = particulator.attributes["volume"].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)


@pytest.mark.xfail(struct.calcsize("P") * 8 == 32, reason="32 bit", strict=False)
def test_coalescence_2_sd(backend_class):
    # Arrange
    s = Settings()
    s.kernel = Golovin(b=1.5e12)
    s.formulae.seed = 0
    steps = [0, 200]
    s.n_sd = 2

    env = Box(dt=s.dt, dv=s.dv)
    builder = Builder(
        n_sd=s.n_sd, backend=backend_class(formulae=s.formulae), environment=env
    )
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
        s.spectrum
    ).sample(s.n_sd)
    builder.add_dynamic(Coalescence(collision_kernel=s.kernel, adaptive=False))
    particulator = builder.build(attributes)

    volumes = {}

    # Act
    for step in steps:
        particulator.run(step - particulator.n_steps)
        volumes[particulator.n_steps] = particulator.attributes["volume"].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)
    assert particulator.attributes.super_droplet_count == 1
