import pytest
import numpy as np
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.builder import Builder
from PySDM.dynamics.collisions.collision import Coalescence
from PySDM.dynamics.collisions.collision_kernels.constantK import ConstantK
from PySDM.physics import si
from PySDM.environments.box import Box


def generate_rand_attr_param(n_sd):
    np.random.seed(12)

    return pytest.param({
        "volume": (3*np.random.random(n_sd) + 1)*si.mm**3,
        "n": np.random.randint(100, 1000, n_sd),
        "fall momentum": (5*np.random.random(n_sd) + 1)*1e-6
    }, id=f"random(n_sd={n_sd})")


@pytest.fixture(params=(
    pytest.param({
        "volume": np.array([si.mm**3, 2*si.mm**3]),
        "n": np.array([1, 1]),
        "fall momentum": np.array([10e-6, 6e-6])
    }, id="two_droplets"),

    pytest.param({
        "volume": np.array([si.mm**3, 2*si.mm**3, 3*si.mm**3]),
        "n": np.array([2, 1, 4]),
        "fall momentum": np.array([10e-6, 6e-6, 4e-6])
    }, id="fixed(n_sd=3)"),

    generate_rand_attr_param(n_sd=100)
))
def default_attributes(request):
    return request.param


def test_fall_velocity_calculation(default_attributes, backend_class):
    """
    Test that fall velocity is the momentum divided by the mass.
    """
    builder = Builder(n_sd=len(default_attributes["n"]), backend=backend_class())
    builder.set_environment(Box(dt=1, dv=1))
    builder.request_attribute("fall velocity")
    particulator = builder.build(
        attributes=default_attributes,
        products=()
    )

    assert np.allclose(
        particulator.attributes["fall velocity"].to_ndarray(),
        particulator.attributes["fall momentum"].to_ndarray() /
        (particulator.formulae.constants.rho_w *
         particulator.attributes["volume"].to_ndarray())
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

    particulator = builder.build(
        attributes=default_attributes,
        products=()
    )

    particulator.run(10)

    total_initial_momentum = (
        default_attributes["fall momentum"]*default_attributes["n"]).sum()
    total_final_momentum = (particulator.attributes["fall momentum"].to_ndarray() *
                            particulator.attributes["n"].to_ndarray()).sum()

    # assert that the total number of droplets changed
    assert not np.sum(particulator.attributes["n"].to_ndarray()) == np.sum(default_attributes["n"])

    # assert that the total momentum is conserved
    assert np.isclose(total_final_momentum, total_initial_momentum)



# def test_conservation_of_momentum(default_attributes, backend_class):
# 
#     builder = Builder(n_sd=len(default_attributes["n"]), backend=backend_class())
# 
#     builder.set_environment(Box(dt=1, dv=1))
#     builder.request_attribute("fall momentum")
# 
#     particulator = builder.build(
#         attributes=default_attributes,
#         products=()
#     )
# 
# 
#     # set up storage for coalescence
#     n_sd = particulator.n_sd
#     n_pairs = n_sd // 2
# 
# 
#     # TODO PROBLEM:
#     # 2-droplets pass with gamma = 1, 100-droplets pass with gamma = 0.001
# 
#     gamma = particulator.PairwiseStorage.from_ndarray(np.array([0.0001] * n_pairs))
#     Ec = particulator.PairwiseStorage.from_ndarray(np.array([0] * n_pairs))
#     coalescence_rate = particulator.Storage.from_ndarray(np.array([0] * n_sd))
#     
#     is_first_in_pair = make_PairIndicator(
#         particulator.backend)(particulator.n_sd)
# 
# 
# 
#     for i in range(10):
# 
#         particulator.collision_coalescence_breakup(
#             enable_breakup=False,
#             gamma=gamma,
#             rand=None,
#             Ec=Ec,
#             Eb=None,
#             fragment_size=None,
#             coalescence_rate=coalescence_rate,
#             breakup_rate=None,
#             breakup_rate_deficit=None,
#             is_first_in_pair=is_first_in_pair,
#             warn_overflows=None,
#             max_multiplicity=None,
#         )
# 
#     # assert that the total number of droplets changed
#     assert not np.sum(particulator.attributes["n"].to_ndarray()) == np.sum(default_attributes["n"])
# 
#     ATTR = "volume"
#     total_final_momentum = (particulator.attributes[ATTR].to_ndarray()*particulator.attributes["n"].to_ndarray()).sum()
#     total_initial_momentum = (default_attributes[ATTR]*default_attributes["n"]).sum()
# 
#     # assert that the total momentum is conserved
#     # assert np.isclose(total_final_momentum, total_initial_momentum)
#     assert abs(total_final_momentum - total_initial_momentum)/total_initial_momentum < 0.01

