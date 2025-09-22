# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics.terminal_velocity import (
    GunnKinzer1949,
    PowerSeries,
    RogersYau,
)
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
            "signed water mass": np.asarray([water_mass]),
            "multiplicity": np.asarray([-1]),
        }
    )

    # act
    with context:
        (v_term,) = particulator.attributes["terminal velocity"].to_ndarray()

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


@pytest.mark.parametrize("ice_variant", ("ColumnarIceCrystal", "IceSphere"))
def test_ice_particle_terminal_velocities_basics(backend_class, ice_variant):
    if backend_class.__name__ == "ThrustRTC":
        pytest.skip()

    # arrange
    water_mass = np.logspace(base=10, start=-16, stop=-7, num=10) * si.kg
    env = Box(dt=None, dv=None)
    formulae_enabling_terminal_velocity_ice_calculation = Formulae(
        particle_shape_and_density="MixedPhaseSpheres",
        terminal_velocity_ice=ice_variant,
    )
    builder = Builder(
        backend=backend_class(formulae_enabling_terminal_velocity_ice_calculation),
        n_sd=len(water_mass),
        environment=env,
    )
    builder.request_attribute("terminal velocity")
    particulator = builder.build(
        attributes={
            "signed water mass": -water_mass,
            "multiplicity": np.ones_like(water_mass),
        }
    )
    atmospheric_settings = [
        {"temperature": 233 * si.kelvin, "pressure": 300 * si.hectopascal},
        {"temperature": 270 * si.kelvin, "pressure": 1000 * si.hectopascal},
    ]
    terminal_velocity = None
    for setting in atmospheric_settings:

        particulator.environment["T"] = setting["temperature"]
        particulator.environment["p"] = setting["pressure"]

        # TODO #1606 the line below should not be needed (auto update when env variables change)
        particulator.attributes.mark_updated("signed water mass")

        # act
        particulator.run(steps=1)

        # assert
        if terminal_velocity is not None:
            assert all(
                terminal_velocity
                != particulator.attributes["terminal velocity"].to_ndarray()
            )

        terminal_velocity = particulator.attributes["terminal velocity"].to_ndarray()

        # assert
        assert all(~np.isnan(terminal_velocity))
        assert all(terminal_velocity > 0.0)
        assert all(np.diff(terminal_velocity) > 0.0)


def test_columnar_ice_crystal_terminal_velocity_against_spichtinger_and_gierens_2009_fig_3(
    backend_class, plot=False
):
    """Fig. 3 in [Spichtinger & Gierens 2009](https://doi.org/10.5194/acp-9-685-2009)"""
    if backend_class.__name__ == "ThrustRTC":
        pytest.skip()
    # arrange
    water_mass = np.logspace(base=10, start=-16, stop=-7, num=10) * si.kg
    terminal_velocity_reference = (
        np.array(
            [
                1.4e-04,
                3.7e-04,
                9.7e-04,
                2.5e-03,
                9.1e-03,
                3.4e-02,
                1.3e-01,
                4.7e-01,
                1.1e00,
                1.9e00,
            ]
        )
        * si.m
        / si.s
    )
    ambient_temperature = 233 * si.K
    ambient_pressure = 300 * si.hectopascal

    env = Box(dt=None, dv=None)
    formulae_enabling_terminal_velocity_ice_calculation = Formulae(
        particle_shape_and_density="MixedPhaseSpheres",
        terminal_velocity_ice="ColumnarIceCrystal",
    )
    builder = Builder(
        backend=backend_class(formulae_enabling_terminal_velocity_ice_calculation),
        n_sd=len(water_mass),
        environment=env,
    )

    builder.request_attribute("terminal velocity")
    particulator = builder.build(
        attributes={
            "signed water mass": -water_mass,
            "multiplicity": np.ones_like(water_mass),
        }
    )

    particulator.environment["T"] = ambient_temperature
    particulator.environment["p"] = ambient_pressure

    # act
    terminal_velocity = particulator.attributes["terminal velocity"].to_ndarray()

    # plot
    plt.xlabel("mass (kg)")
    plt.ylabel("terminal velocity (m/s)")
    plt.xlim(water_mass[0], water_mass[-1])
    plt.xscale("log")
    plt.ylim(1e-4, 1e1)
    plt.yscale("log")
    plt.grid()

    plt.plot(water_mass, terminal_velocity_reference, color="black")
    plt.plot(water_mass, terminal_velocity, color="red")

    if plot:
        plt.show()
    else:
        plt.clf()

    # assert
    np.testing.assert_almost_equal(
        terminal_velocity, terminal_velocity_reference, decimal=1
    )
