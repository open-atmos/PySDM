# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.backends import ThrustRTC
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.environments import Box
from PySDM.formulae import Formulae
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.spectra import Exponential
from PySDM.physics.constants import si

from ...backends_fixture import backend_class

assert hasattr(backend_class, "_pytestfixturefunction")


def check(*, n_part, dv, n_sd, rho, attributes, step):
    check_lwc = 1e-3 * si.kilogram / si.metre**3
    check_ksi = n_part * dv / n_sd

    # multiplicities
    if step == 0:
        np.testing.assert_approx_equal(
            np.amin(attributes["n"]), np.amax(attributes["n"]), 1
        )
        np.testing.assert_approx_equal(attributes["n"][0], check_ksi, 1)

    # liquid water content
    LWC = rho * np.dot(attributes["n"], attributes["volume"]) / dv
    np.testing.assert_approx_equal(LWC, check_lwc, 3)


@pytest.mark.parametrize("croupier", ["local", "global"])
@pytest.mark.parametrize("adaptive", [True, False])
# pylint: disable=redefined-outer-name,too-many-locals
def test_coalescence(backend_class, croupier, adaptive):
    if backend_class == ThrustRTC and croupier == "local":  # TODO #358
        return
    if backend_class == ThrustRTC and adaptive and croupier == "global":  # TODO #329
        return
    # Arrange
    formulae = Formulae(seed=256)
    n_sd = 2**14
    steps = [0, 100, 200]
    X0 = formulae.trivia.volume(radius=30.531e-6)
    n_part = 2**23 / si.metre**3
    dv = 1e6 * si.metres**3
    dt = 1 * si.seconds
    norm_factor = n_part * dv
    rho = 1000 * si.kilogram / si.metre**3

    kernel = Golovin(b=1.5e3)  # [s-1]
    spectrum = Exponential(norm_factor=norm_factor, scale=X0)
    builder = Builder(n_sd=n_sd, backend=backend_class(formulae=formulae))
    builder.set_environment(Box(dt=dt, dv=dv))
    attributes = {}
    attributes["volume"], attributes["n"] = ConstantMultiplicity(spectrum).sample(n_sd)
    builder.add_dynamic(
        Coalescence(collision_kernel=kernel, croupier=croupier, adaptive=adaptive)
    )
    particulator = builder.build(attributes)

    volumes = {}

    # Act
    for step in steps:
        particulator.run(step - particulator.n_steps)
        check(
            n_part=n_part,
            dv=dv,
            n_sd=n_sd,
            rho=rho,
            attributes=particulator.attributes,
            step=step,
        )
        volumes[particulator.n_steps] = particulator.attributes["volume"].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)
