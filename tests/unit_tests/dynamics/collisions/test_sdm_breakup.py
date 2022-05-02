# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.dynamics import Breakup
from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM.dynamics.collisions.collision_kernels import ConstantK
from PySDM.environments import Box
from PySDM.physics import si


class TestSDMBreakup:
    @staticmethod
    @pytest.mark.parametrize(
        "dt",
        [
            1 * si.s,
            10 * si.s,
        ],
    )
    def test_nonadaptive_same_results_regardless_of_dt(dt, backend_class=CPU):
        # Arrange
        attributes = {
            "n": np.asarray([1, 1]),
            "volume": np.asarray([100 * si.um**3, 100 * si.um**3]),
        }
        breakup = Breakup(
            collision_kernel=ConstantK(1 * si.cm**3 / si.s),
            fragmentation_function=AlwaysN(4),
            adaptive=False,
        )
        nsteps = 10

        n_sd = len(attributes["n"])
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=1 * si.cm**3, dt=dt))
        builder.add_dynamic(breakup)
        particulator = builder.build(attributes=attributes, products=())

        # Act
        particulator.run(nsteps)

        # Assert
        assert (particulator.attributes["n"].to_ndarray() > 0).all()
        assert (particulator.attributes["n"].to_ndarray() != attributes["n"]).any()
        assert np.sum(particulator.attributes["n"].to_ndarray()) >= np.sum(
            attributes["n"]
        )
        assert (
            particulator.attributes["n"].to_ndarray() == np.array([1024, 1024])
        ).all()

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": 1.0,
                "rand": 1.0,
            },
            {"gamma": 1.0, "rand": 0.1},
            pytest.param(
                {"gamma": 1.0, "rand": 0.0}, marks=pytest.mark.xfail(strict=True)
            ),
        ],
    )
    def test_single_collision_bounce(params, backend_class=CPU):
        # Arrange
        n_sd = 2
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=np.NaN, dt=np.NaN))
        n_init = [1, 1]
        particulator = builder.build(
            attributes={
                "n": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3, 100 * si.um**3]),
            },
            products=(),
        )

        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(np.array([0.0]))
        general_zeros = particulator.Storage.from_ndarray(np.array([0.0]))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array([params["gamma"]]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array([params["rand"]]))
        n_fragment = particulator.PairwiseStorage.from_ndarray(np.array([4]))
        is_first_in_pair = make_PairIndicator(backend_class)(n_sd)

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=pairwise_zeros,
            n_fragment=n_fragment,
            coalescence_rate=general_zeros,
            breakup_rate=general_zeros,
            breakup_rate_deficit=general_zeros,
            is_first_in_pair=is_first_in_pair,
        )

        # Assert
        assert (particulator.attributes["n"].to_ndarray() == n_init).all()

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": 1.0,
                "rand": 1.0,
                "Eb": 1.0,
                "n_init": [1, 1],
                "is_first_in_pair": [True, False],
            },
            {
                "gamma": 1.0,
                "rand": 1.0,
                "Eb": 1.0,
                "n_init": [2, 1],
                "is_first_in_pair": [True, False],
            },
            {
                "gamma": 1.0,
                "rand": 1.0,
                "Eb": 1.0,
                "n_init": [2, 1, 2],
                "is_first_in_pair": [True, False, False],
            },
            {
                "gamma": 1.0,
                "rand": 1.0,
                "Eb": 1.0,
                "n_init": [2, 1, 2, 1],
                "is_first_in_pair": [True, False, True, False],
            },
        ],
    )
    def test_breakup_counters(params, backend_class=CPU):
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=np.NaN, dt=np.NaN))
        particulator = builder.build(
            attributes={
                "n": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3] * n_sd),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0.0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(
            np.array([params["gamma"]] * n_pairs)
        )
        rand = particulator.PairwiseStorage.from_ndarray(
            np.array([params["rand"]] * n_pairs)
        )
        Eb = particulator.PairwiseStorage.from_ndarray(
            np.array([params["Eb"]] * n_pairs)
        )
        breakup_rate = particulator.Storage.from_ndarray(np.array([0.0]))
        n_fragment = particulator.PairwiseStorage.from_ndarray(np.array([4] * n_pairs))
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"])
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            n_fragment=n_fragment,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=general_zeros,
            is_first_in_pair=is_first_in_pair,
        )

        # Assert
        cell_id = 0
        assert breakup_rate.to_ndarray()[cell_id] == np.sum(
            params["gamma"] * get_smaller_of_pairs(is_first_in_pair, n_init)
        )

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": [1.0],
                "n_init": [1, 1],
                "v_init": [1, 1],
                "n_expected": [2, 2],
                "v_expected": [0.5, 0.5],
                "is_first_in_pair": [True, False],
                "n_fragment": [4],
            },
            {
                "gamma": [2.0],
                "n_init": [20, 4],
                "v_init": [1, 2],
                "n_expected": [4, 36],
                "v_expected": [1, 2 / 3],
                "is_first_in_pair": [True, False],
                "n_fragment": [3],
            },
            {
                "gamma": [2.0],
                "n_init": [1, 1],
                "v_init": [1, 1],
                "n_expected": [4, 4],
                "v_expected": [0.25, 0.25],
                "is_first_in_pair": [True, False],
                "n_fragment": [4],
            },
            {
                "gamma": [2.0],
                "n_init": [3, 1],
                "v_init": [1, 1],
                "n_expected": [8, 2],
                "v_expected": [0.375, 0.5],
                "is_first_in_pair": [True, False],
                "n_fragment": [4],
            },
        ],
    )
    @pytest.mark.parametrize("flag", ("n", "v", "conserve"))
    def test_attribute_update_single_breakup(params, flag, backend_class=CPU):
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=np.NaN, dt=np.NaN))
        particulator = builder.build(
            attributes={
                "n": np.asarray(n_init),
                "volume": np.asarray(params["v_init"]),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        rand = [1.0] * n_pairs
        Eb = [1.0] * n_pairs
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0.0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array(params["gamma"]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0.0]))
        breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0.0]))
        n_fragment = particulator.PairwiseStorage.from_ndarray(
            np.array(params["n_fragment"])
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"])
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            n_fragment=n_fragment,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=breakup_rate_deficit,
            is_first_in_pair=is_first_in_pair,
        )

        # Assert
        {
            "n": lambda: np.testing.assert_array_equal(
                particulator.attributes["n"].to_ndarray(),
                np.array(params["n_expected"]),
            ),
            "v": lambda: np.testing.assert_array_equal(
                particulator.attributes["volume"].to_ndarray(),
                np.array(params["v_expected"]),
            ),
            "conserve": lambda: np.testing.assert_equal(
                np.sum(
                    particulator.attributes["n"].to_ndarray()
                    * particulator.attributes["volume"].to_ndarray()
                ),
                np.sum(np.array(params["n_init"]) * np.array(params["v_init"])),
            ),
        }[flag]()

    @staticmethod
    # @pytest.mark.xfail(strict=True)
    def test_multiplicity_overflow(backend=CPU()):
        # Arrange
        params = {
            "gamma": [100.0],
            "n_init": [1, 1],
            "v_init": [1, 1],
            "is_first_in_pair": [True, False],
            "n_fragment": [1e10],
        }
        n_init = params["n_init"]
        n_sd = len(n_init)
        builder = Builder(n_sd, backend)
        builder.set_environment(Box(dv=np.NaN, dt=np.NaN))
        particulator = builder.build(
            attributes={
                "n": np.asarray(n_init),
                "volume": np.asarray(params["v_init"]),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        rand = [1.0] * n_pairs
        Eb = [1.0] * n_pairs
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0.0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array(params["gamma"]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0.0]))
        breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0.0]))
        n_fragment = particulator.PairwiseStorage.from_ndarray(
            np.array(params["n_fragment"])
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"])
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            n_fragment=n_fragment,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=breakup_rate_deficit,
            is_first_in_pair=is_first_in_pair,
        )
        assert breakup_rate_deficit[0] > 0
        np.testing.assert_equal(
            np.sum(
                particulator.attributes["n"].to_ndarray()
                * particulator.attributes["volume"].to_ndarray()
            ),
            np.sum(np.array(params["n_init"]) * np.array(params["v_init"])),
        )

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": [1.0],
                "n_init": [1, 1],
                "v_init": [1, 1],
                "n_expected": [1, 1],
                "v_expected": [1, 1],
                "is_first_in_pair": [True, False],
                "n_fragment": [1.6],
            },
            {
                "gamma": [2.0],
                "n_init": [20, 4],
                "v_init": [1, 2],
                "n_expected": [6, 25],
                "v_expected": [1, 0.88],
                "is_first_in_pair": [True, False],
                "n_fragment": [2.5],
            },
            {
                "gamma": [2.0],
                "n_init": [2, 1],
                "v_init": [1, 1],
                "n_expected": [3, 2],
                "v_expected": [5 / 9, 2 / 3],
                "is_first_in_pair": [True, False],
                "n_fragment": [2.8],
            },
        ],
    )
    @pytest.mark.parametrize("flag", ("n", "v", "conserve"))
    def test_noninteger_fragments(params, flag, backend_class=CPU):
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=np.NaN, dt=np.NaN))
        particulator = builder.build(
            attributes={
                "n": np.asarray(n_init),
                "volume": np.asarray(params["v_init"]),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        rand = [1.0] * n_pairs
        Eb = [1.0] * n_pairs
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0.0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array(params["gamma"]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0.0]))
        breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0.0]))
        n_fragment = particulator.PairwiseStorage.from_ndarray(
            np.array(params["n_fragment"])
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"])
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            n_fragment=n_fragment,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=breakup_rate_deficit,
            is_first_in_pair=is_first_in_pair,
        )

        # Assert
        {
            "n": lambda: np.testing.assert_array_equal(
                particulator.attributes["n"].to_ndarray(),
                np.array(params["n_expected"]),
            ),
            "v": lambda: np.testing.assert_array_almost_equal(
                particulator.attributes["volume"].to_ndarray(),
                np.array(params["v_expected"]),
                decimal=6,
            ),
            "conserve": lambda: np.testing.assert_almost_equal(
                np.sum(
                    particulator.attributes["n"].to_ndarray()
                    * particulator.attributes["volume"].to_ndarray()
                ),
                np.sum(np.array(params["n_init"]) * np.array(params["v_init"])),
                decimal=6,
            ),
        }[flag]()


def get_smaller_of_pairs(is_first_in_pair, n_init):
    return np.where(
        np.roll(is_first_in_pair.indicator, shift=1), np.asarray(n_init), 0.0
    )
