# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from ast import Break

import numpy as np
import pytest

import PySDM.physics.constants as const
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
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


class TestCollisionProducts:
    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": 1.0,
                "rand": 1.0,
                "enable_breakup": False,
                "cr": 2.0,
                "crd": 0.0,
                "cor": 2.0,
                "br": 0.0,
                "brd": 0.0,
            },
            # {
            #     "gamma": 1.0,
            #     "rand": 1.0,
            #     "enable_breakup": True,
            #     "enable_coalescence": True,
            #     "Ec": 1.0,
            # },
            # {
            #     "gamma": 1.0,
            #     "rand": 1.0,
            #     "enable_breakup": True,
            #     "enable_coalescence": False,
            #     "Ec": 0.0
            # }
        ],
    )
    def test_individual_dynamics_rates(params, backend_class=CPU):
        # Arrange
        n_init = [2, 5]
        n_sd = len(n_init)
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=1 * si.m**3, dt=1 * si.s))
        n_init = [1, 1]

        # pairwise_zeros = particulator.PairwiseStorage.from_ndarray(np.array([0.0]))
        # general_zeros = particulator.Storage.from_ndarray(np.array([0.0]))

        # gamma = particulator.PairwiseStorage.from_ndarray(np.array([params["gamma"]]))
        # rand = particulator.PairwiseStorage.from_ndarray(np.array([params["rand"]]))

        if params["enable_breakup"]:
            if params["enable_coalescence"]:
                dynamic = Collision(
                    collision_kernel=ConstantK(a=1 * si.cm**3 / si.s),
                    coalescence_efficiency=ConstEc(Ec=params["Ec"]),
                    breakup_efficiency=ConstEb(Eb=params["Eb"]),
                    fragmentation_function=AlwaysN(n=params["nf"]),
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
                    collision_kernel=ConstantK(a=1e5 * si.cm**3 / si.s),
                    fragmentation_function=AlwaysN(n=params["nf"]),
                )
                products = (
                    CollisionRatePerGridbox(name="cr"),
                    CollisionRateDeficitPerGridbox(name="crd"),
                    BreakupRatePerGridbox(name="br"),
                    BreakupRateDeficitPerGridbox(name="brd"),
                )
        else:
            dynamic = Coalescence(
                collision_kernel=ConstantK(a=np.inf * si.cm**3 / si.s),
                coalescence_efficiency=ConstEc(Ec=1.0),
            )
            products = (
                CollisionRatePerGridbox(name="cr"),
                CollisionRateDeficitPerGridbox(name="crd"),
                CoalescenceRatePerGridbox(name="cor"),
            )

        builder.add_dynamic(dynamic)
        # is_first_in_pair = make_PairIndicator(backend_class)(n_sd)
        # min_volume = 1 * si.nm**3

        particulator = builder.build(
            attributes={
                "n": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3] * n_sd),
            },
            products=(
                CollisionRatePerGridbox(name="cr"),
                CollisionRateDeficitPerGridbox(name="crd"),
                CoalescenceRatePerGridbox(name="cor"),
                BreakupRatePerGridbox(name="br"),
                BreakupRateDeficitPerGridbox(name="brd"),
            ),
        )

        # Act
        particulator.run(1)

        # Assert
        cr = particulator.products["cr"].get()
        crd = particulator.products["crd"].get()
        assert cr == params["cr"]
        assert crd == params["crd"]
        if params["enable_breakup"]:
            br = particulator.products["br"].get()
            brd = particulator.products["brd"].get()
            assert br == params["br"]
            assert brd == params["brd"]
            if params["enable_coalescence"]:
                cor = particulator.products["cor"].get()
                assert cor == params["cor"]
        else:
            cor = particulator.products["cor"].get()
            assert cor == params["cor"]

    # def test_deficit_rates()

    # def test_rate_sums()
