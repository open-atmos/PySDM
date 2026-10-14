"""
checks if liquid water content remains constant throughout coalescence-only simulation
"""

import numpy as np
import pytest

from PySDM.backends import ThrustRTC
from PySDM import Particulator
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.formulae import Formulae
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.spectra import Exponential
from PySDM.physics.constants import si


def check(*, n_part, dv, n_sd, rho, attributes, step):
    check_lwc = 1e-3 * si.kilogram / si.metre**3
    check_ksi = n_part * dv / n_sd

    # multiplicities
    if step == 0:
        np.testing.assert_approx_equal(
            np.amin(attributes["multiplicity"].to_ndarray()),
            np.amax(attributes["multiplicity"].to_ndarray()),
            significant=1,
        )
        np.testing.assert_approx_equal(
            attributes["multiplicity"].to_ndarray()[0], check_ksi, 1
        )

    # liquid water content
    lwc = (
        rho
        * np.dot(
            attributes["multiplicity"].to_ndarray(), attributes["volume"].to_ndarray()
        )
        / dv
    )
    np.testing.assert_approx_equal(lwc, check_lwc, significant=3)


@pytest.mark.parametrize("croupier", ["local", "global"])
@pytest.mark.parametrize("adaptive", [True, False])
# pylint: disable=too-many-locals
def test_lwc_constant(backend_class_with_jax, croupier, adaptive):
    if backend_class_with_jax == ThrustRTC and croupier == "local":  # TODO #358
        pytest.skip()
    if (
        backend_class_with_jax == ThrustRTC and adaptive and croupier == "global"
    ):  # TODO #329
        pytest.skip()
    # Arrange
    formulae = Formulae(
        seed=256,
        collision_kernel="Golovin",
        constants={"KERNEL_GOLOVIN_b": 1.5e3 / si.s},
    )
    n_sd = 2**14
    steps = [0, 100, 200]
    X0 = formulae.trivia.volume(radius=30.531e-6)
    n_part = 2**23 / si.metre**3
    dv = 1e6 * si.metres**3
    dt = 1 * si.seconds
    norm_factor = n_part * dv
    rho = 1000 * si.kilogram / si.metre**3

    spectrum = Exponential(norm_factor=norm_factor, scale=X0)

    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
        spectrum
    ).sample_deterministic(n_sd)

    particulator = Particulator(
        attributes=attributes,
        n_sd=n_sd,
        environment=Box(
            dt=dt, dv=dv, backend=backend_class_with_jax(formulae=formulae)
        ),
        dynamics=(Coalescence(croupier=croupier, adaptive=adaptive),),
    )

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
