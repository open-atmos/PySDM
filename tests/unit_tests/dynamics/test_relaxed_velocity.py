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
                "n": np.array([1, 2, 3, 4]),
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


def test_small_timescale(default_attributes, backend_class):
    """
    When the fall velocity is initialized to 0 and tau is very small,
    the velocity should quickly approach the terminal velocity
    """

    builder = Builder(n_sd=len(default_attributes["n"]), backend=backend_class())

    builder.set_environment(Box(dt=1, dv=1))

    builder.add_dynamic(RelaxedVelocity(tau=1e-12))

    builder.request_attribute("relative fall velocity")
    builder.request_attribute("terminal velocity")

    default_attributes["relative fall momentum"] = np.zeros_like(
        default_attributes["n"]
    )

    particulator = builder.build(attributes=default_attributes, products=())

    particulator.run(1)

    assert np.allclose(
        particulator.attributes["relative fall velocity"].to_ndarray(),
        particulator.attributes["terminal velocity"].to_ndarray(),
    )


def test_large_timescale(default_attributes, backend_class):
    """
    When the fall velocity is initialized to 0 and tau is very large,
    the velocity should remain 0
    """

    builder = Builder(n_sd=len(default_attributes["n"]), backend=backend_class())

    builder.set_environment(Box(dt=1, dv=1))

    builder.add_dynamic(RelaxedVelocity(tau=1e12))

    builder.request_attribute("relative fall velocity")
    builder.request_attribute("terminal velocity")

    default_attributes["relative fall momentum"] = np.zeros_like(
        default_attributes["n"]
    )

    particulator = builder.build(attributes=default_attributes, products=())

    particulator.run(100)

    assert np.allclose(
        particulator.attributes["relative fall velocity"].to_ndarray(),
        np.zeros_like(default_attributes["n"]),
    )


def test_behavior(default_attributes, backend_class):
    """
    The fall velocity should approach the terminal velocity exponentially
    """

    builder = Builder(n_sd=len(default_attributes["n"]), backend=backend_class())

    builder.set_environment(Box(dt=1, dv=1))

    builder.add_dynamic(RelaxedVelocity(tau=0.5))

    builder.request_attribute("relative fall velocity")
    builder.request_attribute("terminal velocity")

    default_attributes["relative fall momentum"] = np.zeros_like(
        default_attributes["n"]
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
