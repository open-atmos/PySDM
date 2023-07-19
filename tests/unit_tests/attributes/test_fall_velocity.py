import numpy as np
import pytest

from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.builder import Builder
from PySDM.dynamics.collisions.collision import Coalescence
from PySDM.dynamics.collisions.collision_kernels.constantK import ConstantK
from PySDM.environments.box import Box
from PySDM.physics import si


def generate_rand_attr_param(n_sd):
    np.random.seed(12)

    return pytest.param(
        {
            "volume": (3 * np.random.random(n_sd) + 1) * si.mm**3,
            "n": np.random.randint(100, 1000, n_sd),
            "fall momentum": (5 * np.random.random(n_sd) + 1) * 1e-6,
        },
        id=f"random(n_sd={n_sd})",
    )


@pytest.fixture(
    params=(
        pytest.param(
            {
                "volume": np.array([si.mm**3, 2 * si.mm**3]),
                "n": np.array([1, 1]),
                "fall momentum": np.array([10e-6, 6e-6]),
            },
            id="two_droplets",
        ),
        pytest.param(
            {
                "volume": np.array([si.mm**3, 2 * si.mm**3, 3 * si.mm**3]),
                "n": np.array([2, 1, 4]),
                "fall momentum": np.array([10e-6, 6e-6, 4e-6]),
            },
            id="fixed(n_sd=3)",
        ),
        generate_rand_attr_param(n_sd=100),
    )
)
def default_attributes(request):
    return request.param


def test_fall_velocity_calculation(default_attributes, backend_class):
    """
    Test that fall velocity is the momentum divided by the mass.
    """
    builder = Builder(n_sd=len(default_attributes["n"]), backend=backend_class())
    builder.set_environment(Box(dt=1, dv=1))
    builder.request_attribute("fall velocity")
    particulator = builder.build(attributes=default_attributes, products=())

    assert np.allclose(
        particulator.attributes["fall velocity"].to_ndarray(),
        particulator.attributes["fall momentum"].to_ndarray()
        / (
            particulator.formulae.constants.rho_w
            * particulator.attributes["volume"].to_ndarray()
        ),
    )


def test_conservation_of_momentum(default_attributes, backend_class):
    """
    Test that conservation of momentum holds when many super-droplets coalesce
    """
    builder = Builder(n_sd=len(default_attributes["n"]), backend=backend_class())
    builder.set_environment(Box(dt=1, dv=1))
    builder.request_attribute("fall momentum")

    # TODO only works with adaptive=False
    builder.add_dynamic(Coalescence(collision_kernel=ConstantK(a=1), adaptive=True))

    particulator = builder.build(attributes=default_attributes, products=())

    particulator.run(10)

    total_initial_momentum = (
        default_attributes["fall momentum"] * default_attributes["n"]
    ).sum()
    total_final_momentum = (
        particulator.attributes["fall momentum"].to_ndarray()
        * particulator.attributes["n"].to_ndarray()
    ).sum()

    # assert that the total number of droplets changed
    assert not np.sum(particulator.attributes["n"].to_ndarray()) == np.sum(
        default_attributes["n"]
    )

    # assert that the total momentum is conserved
    assert np.isclose(total_final_momentum, total_initial_momentum)
