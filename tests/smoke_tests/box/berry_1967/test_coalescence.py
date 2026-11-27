# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import struct

import numpy as np
import pytest
from PySDM_examples.featured.Berry_1967.settings import Settings
from PySDM.physics import si
from PySDM.backends import ThrustRTC
from PySDM import Particulator
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity


@pytest.mark.parametrize("croupier", ("local", "global"))
@pytest.mark.parametrize("adaptive", (True, False))
@pytest.mark.parametrize("kernel", ("Geometric", "Electric", "Hydrodynamic"))
def test_coalescence(backend_class, kernel, croupier, adaptive):
    if backend_class == ThrustRTC and croupier == "local":
        pytest.skip("TODO #358")
    if backend_class == ThrustRTC and adaptive and croupier == "global":
        pytest.skip("TODO #329")
    if backend_class == ThrustRTC and kernel != "Geometric":
        pytest.skip()

    # Arrange
    s = Settings(
        kernel=kernel if kernel == "Geometric" else "Berry1967",
        constants={
            "Geometric": {},
            "Electric": {
                "KERNEL_BERRY_A": 1,
                "KERNEL_BERRY_B": 1,
                "KERNEL_BERRY_D1": -7,
                "KERNEL_BERRY_D2": 1.78,
                "KERNEL_BERRY_E1": -20.5,
                "KERNEL_BERRY_E2": 1.73,
                "KERNEL_BERRY_F1": 0.26,
                "KERNEL_BERRY_F2": 1.47,
                "KERNEL_BERRY_G1": 1,
                "KERNEL_BERRY_G2": 0.82,
                "KERNEL_BERRY_G3": -0.003,
                "KERNEL_BERRY_Mf": 4.4,
                "KERNEL_BERRY_Mg": 8,
                "KERNEL_BERRY_unit": si.um,
            },
            "Hydrodynamic": {
                "KERNEL_BERRY_A": 1,
                "KERNEL_BERRY_B": 1,
                "KERNEL_BERRY_D1": -27,
                "KERNEL_BERRY_D2": 1.65,
                "KERNEL_BERRY_E1": -58,
                "KERNEL_BERRY_E2": 1.9,
                "KERNEL_BERRY_F1": 15,
                "KERNEL_BERRY_F2": 1.13,
                "KERNEL_BERRY_G1": 16.7,
                "KERNEL_BERRY_G2": 1,
                "KERNEL_BERRY_G3": 0.004,
                "KERNEL_BERRY_Mf": 4,
                "KERNEL_BERRY_Mg": 8,
                "KERNEL_BERRY_unit": si.um,
            },
        }[kernel],
    )
    s.formulae.seed = 0
    steps = [0, 800]

    env = Box(dt=s.dt, dv=s.dv, backend=backend_class(formulae=s.formulae))
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
        s.spectrum
    ).sample_deterministic(s.n_sd)
    particulator = Particulator(
        attributes=attributes,
        n_sd=s.n_sd,
        environment=env,
        dynamics=(Coalescence(croupier=croupier, adaptive=adaptive),),
    )

    volumes = {}

    # Act
    for step in steps:
        particulator.advance(step - particulator.n_steps)
        volumes[particulator.n_steps] = particulator.attributes["volume"].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)


@pytest.mark.xfail(struct.calcsize("P") * 8 == 32, reason="32 bit", strict=False)
def test_coalescence_2_sd(backend_class):
    # Arrange
    s = Settings(kernel="Golovin", constants={"KERNEL_GOLOVIN_b": 1.5e12})
    s.formulae.seed = 0
    steps = [0, 200]
    s.n_sd = 2

    env = Box(dt=s.dt, dv=s.dv, backend=backend_class(formulae=s.formulae))
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
        s.spectrum
    ).sample_deterministic(s.n_sd)
    particulator = Particulator(
        n_sd=s.n_sd,
        environment=env,
        dynamics=(Coalescence(adaptive=False),),
        attributes=attributes,
    )

    volumes = {}

    # Act
    for step in steps:
        particulator.advance(step - particulator.n_steps)
        volumes[particulator.n_steps] = particulator.attributes["volume"].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)
    assert particulator.attributes.super_droplet_count == 1
