# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import numpy as np
import pytest
from PySDM.backends import CPU
from PySDM.dynamics import Breakup
from PySDM.dynamics.collisions.kernels import ConstantK
from PySDM.environments import Box
from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM import Builder
from PySDM.physics import si
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.backends.impl_common.indexed_storage import make_IndexedStorage
from PySDM.backends.impl_common.index import make_Index

class TestSDMBreakup:
    @staticmethod
    @pytest.mark.parametrize("dt", [
        1*si.s,
        10*si.s,
    ])
    def test_nonadaptive_same_results_regardless_of_dt(dt, backend_class = CPU): # TODO move to smoke tests?
        # Arrange
        attributes = {"n": np.asarray([1, 1]), "volume": np.asarray([100*si.um**3, 100*si.um**3])}
        breakup = Breakup(ConstantK(1 * si.cm**3 / si.s), AlwaysN(4), adaptive=False)
        nsteps = 10

        n_sd = len(attributes["n"])
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=1*si.cm**3, dt=dt))
        builder.add_dynamic(breakup)
        particulator = builder.build(attributes = attributes, products = ())

        # Act
        particulator.run(nsteps)

        # Assert
        assert (particulator.attributes['n'].to_ndarray() > 0).all()
        assert (particulator.attributes['n'].to_ndarray() != attributes['n']).any()
        assert (np.sum(particulator.attributes['n'].to_ndarray()) >= np.sum(attributes['n']))
        assert (particulator.attributes['n'].to_ndarray() == np.array([1024, 1024])).all()

    @staticmethod
    @pytest.mark.parametrize("params", [
        {"gamma": 1.0, "rand": 1.0,},
        {"gamma": 1.0, "rand": 0.1},
        pytest.param({"gamma": 1.0, "rand": 0.0}, marks=pytest.mark.xfail(strict=True))
        ])
    def test_single_collision_bounce(params, backend_class = CPU):
        # Arrange
        n_sd = 2
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=np.NaN, dt=np.NaN))
        n_init = [1, 1]
        particulator = builder.build(attributes = {
                "n": np.asarray(n_init),
                "volume": np.asarray([100*si.um**3, 100*si.um**3])
            }, products = ())

        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(np.array([0.0]))
        dropwise_zeros = particulator.Storage.from_ndarray(np.array([0.0, 0.0]))
        general_zeros = particulator.Storage.from_ndarray(np.array([0.0]))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array([params["gamma"]]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array([params["rand"]]))
        n_fragment = particulator.PairwiseStorage.from_ndarray(np.array([4]))
        is_first_in_pair = make_PairIndicator(backend_class)(n_sd)

        # Act
        particulator.collision(gamma = gamma, rand = rand, Ec = pairwise_zeros, Eb = pairwise_zeros,
                n_fragment = n_fragment, coalescence_rate = general_zeros, breakup_rate = general_zeros, is_first_in_pair = is_first_in_pair)

        # Assert
        assert (particulator.attributes['n'].to_ndarray() == n_init).all()

    @staticmethod
    @pytest.mark.parametrize("params", [
        {"gamma": 1.0, "rand": 1.0, "Eb": 1.0, "n_init": [1, 1], "is_first_in_pair": [True, False]},
        {"gamma": 1.0, "rand": 1.0, "Eb": 1.0, "n_init": [2, 1], "is_first_in_pair": [True, False]},
        {"gamma": 1.0, "rand": 1.0, "Eb": 1.0, "n_init": [2, 1, 2], "is_first_in_pair": [True, False, False]},
        {"gamma": 1.0, "rand": 1.0, "Eb": 1.0, "n_init": [2, 1, 2, 1], "is_first_in_pair": [True, False, True, False]},
        ])
    def test_breakup_counters(params, backend_class = CPU):
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=np.NaN, dt=np.NaN))
        particulator = builder.build(attributes = {
                "n": np.asarray(n_init),
                "volume": np.asarray([100*si.um**3] * n_sd)
            }, products = ())

        n_pairs = n_sd // 2 
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(np.array([0.0] * n_pairs))
        general_zeros = particulator.Storage.from_ndarray(np.array([0.0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array([params["gamma"]] * n_pairs))
        rand = particulator.PairwiseStorage.from_ndarray(np.array([params["rand"]] * n_pairs))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array([params["Eb"]] * n_pairs))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0.0]))
        n_fragment = particulator.PairwiseStorage.from_ndarray(np.array([4] * n_pairs))
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"]))

        # Act
        particulator.collision(gamma = gamma, rand = rand, Ec = pairwise_zeros, Eb = Eb,
                n_fragment = n_fragment, coalescence_rate = general_zeros, breakup_rate = breakup_rate, is_first_in_pair = is_first_in_pair)

        # Assert
        cell_id = 0
        assert breakup_rate.to_ndarray()[cell_id] == np.sum(params["gamma"] * np.where(np.roll(is_first_in_pair.indicator, shift = 1), np.asarray(n_init), 0.0))

    @staticmethod
    @pytest.mark.parametrize("params", [
        {"gamma": 1.0, "rand": 1.0, "Eb": 1.0, "n_init": [1, 1], "is_first_in_pair": [True, False]},
        {"gamma": 1.0, "rand": 1.0, "Eb": 1.0, "n_init": [2, 1], "is_first_in_pair": [True, False]},
        {"gamma": 1.0, "rand": 1.0, "Eb": 1.0, "n_init": [2, 1, 2], "is_first_in_pair": [True, False, False]},
        {"gamma": 1.0, "rand": 1.0, "Eb": 1.0, "n_init": [2, 1, 2, 1], "is_first_in_pair": [True, False, True, False]},
        ])
    def test_single_collision_breakup(params, backend_class = CPU):
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        builder = Builder(n_sd, backend_class())
        builder.set_environment(Box(dv=np.NaN, dt=np.NaN))
        particulator = builder.build(attributes = {
                "n": np.asarray(n_init),
                "volume": np.asarray([100*si.um**3] * n_sd)
            }, products = ())

        n_pairs = n_sd // 2 
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(np.array([0.0] * n_pairs))
        general_zeros = particulator.Storage.from_ndarray(np.array([0.0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array([params["gamma"]] * n_pairs))
        rand = particulator.PairwiseStorage.from_ndarray(np.array([params["rand"]] * n_pairs))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array([params["Eb"]] * n_pairs))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0.0]))
        n_fragment = particulator.PairwiseStorage.from_ndarray(np.array([4] * n_pairs))
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"]))

        # Act
        particulator.collision(gamma = gamma, rand = rand, Ec = pairwise_zeros, Eb = Eb,
                n_fragment = n_fragment, coalescence_rate = general_zeros, breakup_rate = breakup_rate, is_first_in_pair = is_first_in_pair)

        # Assert
        cell_id = 0
        assert breakup_rate.to_ndarray()[cell_id] == np.sum(params["gamma"] * np.where(np.roll(is_first_in_pair.indicator, shift = 1), np.asarray(n_init), 0.0))
