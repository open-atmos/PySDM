"""
Test `PySDM.attributes.physics.relative_fall_velocity.RelativeFallVelocity`
and `PySDM.attributes.physics.relative_fall_velocity.RelativeFallMomentum` attributes
"""

import numpy as np
import pytest

from PySDM.attributes.physics import RelativeFallVelocity, TerminalVelocity
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence, RelaxedVelocity
from PySDM.dynamics.collisions.collision_kernels.constantK import ConstantK
from PySDM.environments.box import Box
from PySDM.physics import si


def generate_rand_attr_param(n_sd):
    np.random.seed(12)

    return pytest.param(
        {
            "volume": (3 * np.random.random(n_sd) + 1) * si.mm**3,
            "multiplicity": np.random.randint(100, 1000, n_sd),
            "relative fall momentum": (5 * np.random.random(n_sd) + 1) * 1e-6,
        },
        id=f"random(n_sd={n_sd})",
    )


@pytest.fixture(
    name="default_attributes",
    params=(
        pytest.param(
            {
                "volume": np.array([si.mm**3, 2 * si.mm**3]),
                "multiplicity": np.array([1, 1]),
                "relative fall momentum": np.array([10e-6, 6e-6]),
            },
            id="two_droplets",
        ),
        pytest.param(
            {
                "volume": np.array([si.mm**3, 2 * si.mm**3, 3 * si.mm**3]),
                "multiplicity": np.array([2, 1, 4]),
                "relative fall momentum": np.array([10e-6, 6e-6, 4e-6]),
            },
            id="fixed(n_sd=3)",
        ),
        generate_rand_attr_param(n_sd=100),
    ),
)
def default_attributes_fixture(request):
    return request.param


def test_fall_velocity_calculation(default_attributes, backend_instance):
    """
    Test that fall velocity is the momentum divided by the mass.
    """
    env = Box(dt=1, dv=1)
    builder = Builder(
        n_sd=len(default_attributes["multiplicity"]),
        backend=backend_instance,
        environment=env,
    )

    # needed to use relative fall velocity instead of terminal
    # velocity behind the scenes
    builder.add_dynamic(RelaxedVelocity())

    builder.request_attribute("relative fall velocity")

    particulator = builder.build(attributes=default_attributes, products=())

    assert np.allclose(
        particulator.attributes["relative fall velocity"].to_ndarray(),
        particulator.attributes["relative fall momentum"].to_ndarray()
        / (particulator.attributes["water mass"].to_ndarray()),
    )


def test_conservation_of_momentum(default_attributes, backend_instance):
    """
    Test that conservation of momentum holds when many super-droplets coalesce
    """
    env = Box(dt=1, dv=1)
    builder = Builder(
        n_sd=len(default_attributes["multiplicity"]),
        backend=backend_instance,
        environment=env,
    )

    # add and remove relaxed velocity to prevent warning
    builder.add_dynamic(RelaxedVelocity())

    builder.request_attribute("relative fall momentum")

    builder.particulator.dynamics.pop("RelaxedVelocity")

    builder.add_dynamic(Coalescence(collision_kernel=ConstantK(a=1), adaptive=False))

    particulator = builder.build(attributes=default_attributes, products=())

    particulator.run(2)

    total_initial_momentum = (
        default_attributes["relative fall momentum"]
        * default_attributes["multiplicity"]
    ).sum()

    total_final_momentum = (
        particulator.attributes["relative fall momentum"].to_ndarray()
        * particulator.attributes["multiplicity"].to_ndarray()
    ).sum()

    # assert that the total number of droplets changed
    assert np.sum(particulator.attributes["multiplicity"].to_ndarray()) != np.sum(
        default_attributes["multiplicity"]
    )

    # assert that the total momentum is conserved
    assert np.isclose(total_final_momentum, total_initial_momentum)


def test_attribute_selection(backend_instance):
    """
    Test that the correct velocity attribute is selected by the mapper.
    `PySDM.attributes.physics.relative_fall_velocity.RelativeFallVelocity`
    should only be selected when `PySDM.dynamics.RelaxedVelocity` dynamic exists.
    """
    env = Box(dt=1, dv=1)
    builder_no_relax = Builder(n_sd=1, backend=backend_instance, environment=env)
    builder_no_relax.request_attribute("relative fall velocity")

    # with no RelaxedVelocity, the builder should use TerminalVelocity
    assert isinstance(
        builder_no_relax.req_attr["relative fall velocity"], TerminalVelocity
    )
    env = Box(dt=1, dv=1)
    builder = Builder(n_sd=1, backend=backend_instance, environment=env)
    builder.add_dynamic(RelaxedVelocity())
    builder.request_attribute("relative fall velocity")

    # with RelaxedVelocity, the builder should use RelativeFallVelocity
    assert isinstance(builder.req_attr["relative fall velocity"], RelativeFallVelocity)

    # requesting momentum with no dynamic issues a warning
    env = Box(dt=1, dv=1)
    builder = Builder(n_sd=1, backend=backend_instance, environment=env)
    with pytest.warns(UserWarning):
        builder.request_attribute("relative fall momentum")
