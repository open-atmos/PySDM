# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.dynamics import Breakup, Coalescence, Collision
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision_kernels import ConstantK
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products import (
    BreakupRateDeficitPerGridbox,
    BreakupRatePerGridbox,
    CoalescenceRatePerGridbox,
    CollisionRateDeficitPerGridbox,
    CollisionRatePerGridbox,
)

ENV_ARGS = {"dv": 1 * si.m**3, "dt": 1 * si.s}
RHO_DRY = 1 * si.kg / si.m**3


class TestCollisionProducts:
    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {  # coalescence only
                "enable_breakup": False,
                "cr": 4.0,
                "crd": 6.0,
                "cor": 4.0,
            },
            {  # breakup and coalescence
                "enable_breakup": True,
                "enable_coalescence": True,
                "Ec": 1.0,
                "Eb": 1.0,
                "nf": np.pi,
                "cr": 4.0,
                "crd": 6.0,
                "cor": 4.0,
                "br": 0.0,
                "brd": 0.0,
            },
            {  # breakup and coalescence
                "enable_breakup": True,
                "enable_coalescence": True,
                "Ec": 0.0,
                "Eb": 1.0,
                "nf": 2.0,
                "cr": 4.0,
                "crd": 6.0,
                "cor": 0.0,
                "br": 2.0,
                "brd": 2.0,
            },
            {  # breakup only
                "enable_breakup": True,
                "enable_coalescence": False,
                "nf": 2.0,
                "cr": 4.0,
                "crd": 6.0,
                "br": 2.0,
                "brd": 2.0,
            },
        ],
    )
    def test_individual_dynamics_rates_nonadaptive(params, backend_instance):
        if (
            backend_instance.__class__.__name__ == "ThrustRTC"
            and params["enable_breakup"]
        ):
            pytest.skip("# TODO #744")

        # Arrange
        n_init = [5, 2]
        n_sd = len(n_init)

        env = Box(**ENV_ARGS)
        builder = Builder(n_sd, backend_instance, environment=env)

        dynamic, products = _get_dynamics_and_products(params, adaptive=False)
        builder.add_dynamic(dynamic)

        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3] * n_sd),
            },
            products=products,
        )
        env["rhod"] = RHO_DRY

        # Act
        particulator.run(1)

        # Assert
        assert particulator.products["cr"].get()[0] == params["cr"]
        assert particulator.products["crd"].get()[0] == params["crd"]
        if params["enable_breakup"]:
            assert particulator.products["br"].get()[0] == params["br"]
            assert particulator.products["brd"].get()[0] == params["brd"]
            if params["enable_coalescence"]:
                assert particulator.products["cor"].get()[0] == params["cor"]
        else:
            assert particulator.products["cor"].get()[0] == params["cor"]

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {  # coalescence only
                "enable_breakup": False,
            },
            {  # breakup and coalescence
                "enable_breakup": True,
                "enable_coalescence": True,
                "Ec": 1.0,
                "Eb": 1.0,
                "nf": np.pi,
            },
            {  # breakup only
                "enable_breakup": True,
                "enable_coalescence": False,
                "nf": np.pi,
            },
        ],
    )
    @pytest.mark.parametrize(
        "n_init",
        [[5, 2], [1, 2, 3, 4], [3, 7] * 10],
    )
    def test_no_collision_deficits_when_adaptive(params, n_init, backend_class=CPU):
        # Arrange
        n_sd = len(n_init)
        env = Box(**ENV_ARGS)
        builder = Builder(n_sd, backend_class(), environment=env)

        dynamic, _ = _get_dynamics_and_products(
            params, adaptive=True, kernel_a=1e4 * si.cm**3 / si.s
        )
        builder.add_dynamic(dynamic)

        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3] * n_sd),
            },
            products=(CollisionRateDeficitPerGridbox(name="crd"),),
        )
        env["rhod"] = RHO_DRY

        # Act
        particulator.run(1)

        # Assert
        np.testing.assert_equal(
            particulator.products["crd"].get()[0], np.asarray([0.0] * (n_sd // 2))
        )

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {  # breakup and coalescence
                "enable_breakup": True,
                "enable_coalescence": True,
                "Ec": 0.1,
                "Eb": 1.0,
                "nf": np.pi,
            },
            {  # breakup only
                "enable_breakup": True,
                "enable_coalescence": False,
                "nf": np.pi,
            },
        ],
    )
    def test_breakup_deficits_when_adaptive(params, backend_class=CPU):
        # Arrange
        n_init = [7, 353]
        n_sd = len(n_init)
        env = Box(**ENV_ARGS)
        builder = Builder(n_sd, backend_class(), environment=env)

        dynamic, _ = _get_dynamics_and_products(params, adaptive=True)
        builder.add_dynamic(dynamic)

        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3] * n_sd),
            },
            products=(
                CollisionRateDeficitPerGridbox(name="crd"),
                BreakupRateDeficitPerGridbox(name="brd"),
            ),
        )
        env["rhod"] = RHO_DRY

        # Act
        particulator.run(1)

        # Assert
        assert (
            particulator.products["brd"].get()[0] > np.asarray([0.0] * (n_sd // 2))
        ).all()

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {  # breakup and coalescence
                "enable_breakup": True,
                "enable_coalescence": True,
                "Ec": 0.1,
                "Eb": 1.0,
                "nf": np.pi,
            },
            {  # breakup only
                "enable_breakup": True,
                "enable_coalescence": False,
                "nf": np.pi,
            },
        ],
    )
    def test_no_breakup_deficits_when_while_loop(params, backend_class=CPU):
        # Arrange
        n_init = [7, 353]
        n_sd = len(n_init)
        env = Box(**ENV_ARGS)
        builder = Builder(
            n_sd, backend_class(Formulae(handle_all_breakups=True)), environment=env
        )

        dynamic, _ = _get_dynamics_and_products(
            params, adaptive=True, kernel_a=1e4 * si.cm**3 / si.s
        )
        builder.add_dynamic(dynamic)

        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3] * n_sd),
            },
            products=(
                BreakupRatePerGridbox(name="br"),
                BreakupRateDeficitPerGridbox(name="brd"),
            ),
        )
        env["rhod"] = RHO_DRY

        # Act
        particulator.run(1)

        # Assert
        assert (
            particulator.products["br"].get()[0] > np.asarray([0.0] * (n_sd // 2))
        ).all()
        assert (
            particulator.products["brd"].get()[0] == np.asarray([0.0] * (n_sd // 2))
        ).all()

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {  # coalescence only
                "enable_breakup": False,
            },
            {  # breakup and coalescence
                "enable_breakup": True,
                "enable_coalescence": True,
                "Ec": 0.1,
                "Eb": 1.0,
                "nf": np.pi,
            },
            {  # breakup only
                "enable_breakup": True,
                "enable_coalescence": False,
                "nf": np.pi,
            },
        ],
    )
    def test_rate_sums_single_cell(params, backend_class=CPU):
        # Arrange
        n_init = [7, 353]
        n_sd = len(n_init)
        env = Box(**ENV_ARGS)
        builder = Builder(n_sd, backend_class(), environment=env)

        dynamic, products = _get_dynamics_and_products(params, adaptive=False)
        builder.add_dynamic(dynamic)

        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3] * n_sd),
            },
            products=products,
        )
        env["rhod"] = RHO_DRY

        # Act
        particulator.run(1)
        rhs_sum = _get_product_component_sums(params, particulator.products)

        # Assert
        assert (particulator.products["cr"].get()[0] == rhs_sum).all()


def _get_dynamics_and_products(params, adaptive, kernel_a=1e6 * si.cm**3 / si.s):
    kernel = ConstantK(a=kernel_a)
    if params["enable_breakup"]:
        if params["enable_coalescence"]:
            dynamic = Collision(
                collision_kernel=kernel,
                coalescence_efficiency=ConstEc(Ec=params["Ec"]),
                breakup_efficiency=ConstEb(Eb=params["Eb"]),
                fragmentation_function=AlwaysN(n=params["nf"]),
                adaptive=adaptive,
            )
            products = (
                CollisionRatePerGridbox(name="cr"),
                CollisionRateDeficitPerGridbox(name="crd"),
                CoalescenceRatePerGridbox(name="cor"),
                BreakupRatePerGridbox(name="br"),
                BreakupRateDeficitPerGridbox(name="brd"),
            )
        else:
            dynamic = Breakup(
                collision_kernel=kernel,
                fragmentation_function=AlwaysN(n=params["nf"]),
                adaptive=adaptive,
            )
            products = (
                CollisionRatePerGridbox(name="cr"),
                CollisionRateDeficitPerGridbox(name="crd"),
                BreakupRatePerGridbox(name="br"),
                BreakupRateDeficitPerGridbox(name="brd"),
            )
    else:
        dynamic = Coalescence(
            collision_kernel=kernel,
            coalescence_efficiency=ConstEc(Ec=1.0),
            adaptive=adaptive,
        )
        products = (
            CollisionRatePerGridbox(name="cr"),
            CollisionRateDeficitPerGridbox(name="crd"),
            CoalescenceRatePerGridbox(name="cor"),
        )
    return (dynamic, products)


def _get_product_component_sums(params, products):
    if params["enable_breakup"]:
        product_sum = products["br"].get()[0] + products["brd"].get()[0]
        if params["enable_coalescence"]:
            product_sum += products["cor"].get()[0]
    else:
        product_sum = products["cor"].get()[0]

    return product_sum


def _get_product_rate_diffs_multicell(output, breakup):
    product_sum = output["coalescence_rate"]
    if breakup:
        product_sum += output["breakup_rate"]
        product_sum += output["breakup_deficit"]

    product_sum -= output["collision_rate"]

    return product_sum
