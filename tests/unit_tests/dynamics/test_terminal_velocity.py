# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics.terminal_velocity import GunnKinzer1949, PowerSeries, RogersYau
from PySDM.environments import Box
from PySDM.physics import constants as const
from PySDM.physics import si
from tests.unit_tests.dummy_particulator import DummyParticulator


def test_approximation(backend_class, plot=False):
    r = (
        np.array(
            [0.078, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
        )
        * const.si.mm
        / 2
    )
    particulator = DummyParticulator(
        backend_class, n_sd=len(r), formulae=Formulae(terminal_velocity="RogersYau")
    )
    r = particulator.backend.Storage.from_ndarray(r)
    u = (
        np.array([18, 27, 72, 117, 162, 206, 247, 287, 327, 367, 403, 464, 517, 565])
        / 100
    )
    u_term_ry = particulator.backend.Storage.empty((len(u),), float)
    RogersYau(particulator=particulator)(u_term_ry, r)

    u_term_inter = particulator.backend.Storage.from_ndarray(u_term_ry.to_ndarray())
    GunnKinzer1949(particulator)(u_term_inter, r)

    assert np.mean((u - u_term_ry) ** 2) < 2e-2
    assert np.mean((u - u_term_inter) ** 2) < 1e-6

    if plot:
        plt.plot(r, u_term_ry)
        plt.plot(r, u_term_inter)
        plt.plot(r, u)
        plt.grid()
        plt.show()


@pytest.mark.parametrize(
    "variant, water_mass, exception_context, expected_v_term",
    (
        ("GunnKinzer1949", 0 * si.g, None, 0),
        ("GunnKinzer1949", 1e10 * si.kg, pytest.raises(ValueError, match="Radii"), -1),
        ("RogersYau", 0 * si.g, None, 0),
        ("TpDependent", 0 * si.g, None, 0),
    ),
)
def test_terminal_velocity_boundary_values(
    variant, backend_class, water_mass, exception_context, expected_v_term
):
    if variant == "TpDependent":
        pytest.skip()  # TODO #348

    # arrange
    if exception_context is None:
        context = nullcontext()
    else:
        context = exception_context

    formulae = Formulae(terminal_velocity=variant)
    env = Box(dv=np.nan, dt=np.nan)
    builder = Builder(n_sd=1, backend=backend_class(formulae), environment=env)
    builder.request_attribute("terminal velocity")
    particulator = builder.build(
        attributes={
            "water mass": np.asarray([water_mass]),
            "multiplicity": np.asarray([-1]),
        }
    )

    # act
    with context:
        v_term = particulator.attributes["terminal velocity"].to_ndarray()

        # assert
        np.testing.assert_approx_equal(v_term, expected_v_term)


@pytest.mark.parametrize(
    "prefactors, powers",
    [
        ([2.0], [1 / 6]),
        ([2.0, 1.0], [1 / 6, 1 / 8]),
        pytest.param([1.0], [1 / 6, 1 / 8], marks=pytest.mark.xfail(strict=True)),
    ],
)
def test_power_series(backend_class, prefactors, powers):
    r = np.array([0.01, 0.1, 1.0]) * const.si.mm / 2
    particulator = DummyParticulator(backend_class, n_sd=len(r))
    u = np.zeros_like(r)
    for j, pref in enumerate(prefactors):
        u = u + pref * 4 / 3 * const.PI * (r ** (powers[j] * 3))
    r = particulator.backend.Storage.from_ndarray(r)

    u_term_ps = particulator.backend.Storage.empty((len(u),), float)
    PowerSeries(particulator=particulator, prefactors=prefactors, powers=powers)(
        u_term_ps, r
    )

    u_term_true = particulator.backend.Storage.from_ndarray(u)

    np.testing.assert_array_almost_equal(u, u_term_true)
