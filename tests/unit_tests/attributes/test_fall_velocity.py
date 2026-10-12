"""
Test `PySDM.attributes.physics.relative_fall_velocity.RelativeFallVelocity`
and `PySDM.attributes.physics.relative_fall_velocity.RelativeFallMomentum` attributes
"""

import numpy as np
import pytest

from PySDM.attributes.physics import RelativeFallVelocity, TerminalVelocity
from PySDM import Particulator
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


class TestFallVelocity:
    @staticmethod
    def test_fall_velocity_calculation(default_attributes, backend_instance):
        """
        Test that fall velocity is the momentum divided by the mass.
        """
        env = Box(dt=1, dv=1, backend=backend_instance)

        particulator = Particulator(
            attributes=default_attributes,
            products=(),
            n_sd=len(default_attributes["multiplicity"]),
            environment=env,
            # needed to use relative fall velocity instead of terminal
            # velocity behind the scenes
            dynamics=(RelaxedVelocity(),),
            requested_attributes=("relative fall velocity",),
        )

        assert np.allclose(
            particulator.attributes["relative fall velocity"].to_ndarray(),
            particulator.attributes["relative fall momentum"].to_ndarray()
            / (particulator.attributes["signed water mass"].to_ndarray()),
        )

    @staticmethod
    def test_conservation_of_momentum(default_attributes, backend_instance):
        """
        Test that conservation of momentum holds when many super-droplets coalesce
        """
        env = Box(dt=1, dv=1, backend=backend_instance)

        particulator = Particulator(
            n_sd=len(default_attributes["multiplicity"]),
            environment=env,
            attributes=default_attributes,
            requested_attributes=("relative fall momentum",),
            products=(),
            dynamics=(
                RelaxedVelocity(),
                Coalescence(collision_kernel=ConstantK(a=1), adaptive=False),
            ),
        )

        particulator.dynamics.pop("RelaxedVelocity")

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

    @staticmethod
    def test_attribute_selection(backend_instance):
        """
        Test that the correct velocity attribute is selected by the mapper.
        `PySDM.attributes.physics.relative_fall_velocity.RelativeFallVelocity`
        should only be selected when `PySDM.dynamics.RelaxedVelocity` dynamic exists.
        """
        env = Box(dt=1, dv=1, backend=backend_instance)
        particulator_no_relax = Particulator(
            requested_attributes=("relative fall velocity",),
            n_sd=1,
            environment=env,
            attributes={"multiplicity": np.ones(1), "water mass": np.zeros(1)},
            products=(),
        )

        # with no RelaxedVelocity, the builder should use TerminalVelocity
        assert isinstance(
            particulator_no_relax.req_attr["relative fall velocity"], TerminalVelocity
        )
        env = Box(dt=1, dv=1, backend=backend_instance)

        particulator = Particulator(
            requested_attributes=("relative fall velocity",),
            n_sd=1,
            environment=env,
            dynamics=(RelaxedVelocity(),),
            attributes={
                "multiplicity": np.ones(1),
                "signed water mass": np.zeros(1),
                "relative fall momentum": np.zeros(1),
            },
            products=(),
        )

        # with RelaxedVelocity, the builder should use RelativeFallVelocity
        assert isinstance(
            particulator.req_attr["relative fall velocity"], RelativeFallVelocity
        )

        # requesting momentum with no dynamic issues a warning
        env = Box(dt=1, dv=1, backend=backend_instance)
        with pytest.warns(UserWarning):
            _ = Particulator(
                n_sd=1,
                environment=env,
                requested_attributes=("relative fall momentum",),
                attributes={
                    "multiplicity": np.ones(1),
                    "signed water mass": np.zeros(1),
                },
                products=(),
            )
