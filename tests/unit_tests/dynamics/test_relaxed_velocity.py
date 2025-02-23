"""
Test `PySDM.dynamics.RelaxedVelocity` dynamic
"""

import numpy as np
import pytest

from PySDM.builder import Builder
from PySDM.dynamics import RelaxedVelocity
from PySDM.environments.box import Box
from PySDM.physics import si


@pytest.fixture(
    name="default_attributes",
    params=(
        pytest.param(
            {
                "multiplicity": np.array([1, 2, 3, 4]),
                "volume": np.array(
                    [
                        1 * si.mm**3,
                        0.1 * si.mm**3,
                        1 * si.mm**3,
                        0.05 * si.mm**3,
                    ]
                ),
            },
            id="",
        ),
    ),
)
def default_attributes_fixture(request):
    return request.param


@pytest.fixture(
    name="constant_timescale",
    params=(True, False),
)
def constant_timescale_fixture(request):
    return request.param


def test_small_timescale(default_attributes, constant_timescale, backend_instance):
    """
    When the fall velocity is initialized to 0 and relaxation is very quick,
    the velocity should quickly approach the terminal velocity
    """

    env = Box(dt=1, dv=1)
    builder = Builder(
        n_sd=len(default_attributes["multiplicity"]),
        backend=backend_instance,
        environment=env,
    )

    builder.add_dynamic(RelaxedVelocity(c=1e-12, constant=constant_timescale))

    builder.request_attribute("relative fall velocity")
    builder.request_attribute("terminal velocity")

    default_attributes["relative fall momentum"] = np.zeros_like(
        default_attributes["multiplicity"]
    )

    particulator = builder.build(attributes=default_attributes, products=())

    particulator.run(1)

    assert np.allclose(
        particulator.attributes["relative fall velocity"].to_ndarray(),
        particulator.attributes["terminal velocity"].to_ndarray(),
    )


def test_large_timescale(default_attributes, constant_timescale, backend_instance):
    """
    When the fall velocity is initialized to 0 and relaxation is very slow,
    the velocity should remain 0
    """

    env = Box(dt=1, dv=1)
    builder = Builder(
        n_sd=len(default_attributes["multiplicity"]),
        backend=backend_instance,
        environment=env,
    )

    builder.add_dynamic(RelaxedVelocity(c=1e15, constant=constant_timescale))

    builder.request_attribute("relative fall velocity")
    builder.request_attribute("terminal velocity")

    default_attributes["relative fall momentum"] = np.zeros_like(
        default_attributes["multiplicity"]
    )

    particulator = builder.build(attributes=default_attributes, products=())

    particulator.run(100)

    assert np.allclose(
        particulator.attributes["relative fall velocity"].to_ndarray(),
        np.zeros_like(default_attributes["multiplicity"]),
    )


def test_behavior(default_attributes, constant_timescale, backend_instance):
    """
    The fall velocity should approach the terminal velocity exponentially
    """

    env = Box(dt=1, dv=1)
    builder = Builder(
        n_sd=len(default_attributes["multiplicity"]),
        backend=backend_instance,
        environment=env,
    )

    # relaxation happens too quickly unless c is high enough
    builder.add_dynamic(RelaxedVelocity(c=100, constant=constant_timescale))

    builder.request_attribute("relative fall velocity")
    builder.request_attribute("terminal velocity")

    default_attributes["relative fall momentum"] = np.zeros_like(
        default_attributes["multiplicity"]
    )

    particulator = builder.build(attributes=default_attributes, products=())

    particulator.run(1)
    delta_v1 = (
        particulator.attributes["terminal velocity"].to_ndarray()
        - particulator.attributes["relative fall velocity"].to_ndarray()
    )

    particulator.run(1)
    delta_v2 = (
        particulator.attributes["terminal velocity"].to_ndarray()
        - particulator.attributes["relative fall velocity"].to_ndarray()
    )

    particulator.run(1)
    delta_v3 = (
        particulator.attributes["terminal velocity"].to_ndarray()
        - particulator.attributes["relative fall velocity"].to_ndarray()
    )

    # for an exponential decay, the ratio should be roughly constant using constant timesteps
    assert (np.abs(delta_v1 / delta_v2 - delta_v2 / delta_v3) < 0.01).all()


@pytest.mark.parametrize("c", [0.1, 10])
def test_timescale(default_attributes, c, constant_timescale, backend_instance):
    """
    The non-constant timescale should be proportional to the sqrt of the radius. The
    proportionality constant should be the parameter for the dynamic.

    The constant timescale should be constant.
    """
    env = Box(dt=1, dv=1)
    builder = Builder(
        n_sd=len(default_attributes["multiplicity"]),
        backend=backend_instance,
        environment=env,
    )

    dyn = RelaxedVelocity(c=c, constant=constant_timescale)
    builder.add_dynamic(dyn)

    default_attributes["relative fall momentum"] = np.zeros_like(
        default_attributes["multiplicity"]
    )

    particulator = builder.build(attributes=default_attributes, products=())
    sqrt_radius_attr = builder.get_attribute("square root of radius")

    tau_storage = particulator.Storage.empty(
        default_attributes["multiplicity"].shape, dtype=float
    )
    dyn.calculate_tau(tau_storage, sqrt_radius_attr.get())

    # expected_c should be whatever c was set to in the dynamic
    if not constant_timescale:
        expected_c = tau_storage.to_ndarray() / sqrt_radius_attr.get().to_ndarray()
    else:
        expected_c = tau_storage.to_ndarray()

    assert np.allclose(expected_c, c)
