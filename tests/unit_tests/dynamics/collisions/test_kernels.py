# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics.collisions.collision_kernels import Golovin, SimpleGeometric
from PySDM.environments import Box
from PySDM.formulae import Formulae


class TestKernels:
    @staticmethod
    @pytest.mark.parametrize(
        "x", [pytest.param(5e-10), pytest.param(np.full(10, 5e-10))]
    )
    def test_golovin_analytic_solution_underflow(x):
        # Arrange
        formulae = Formulae()
        b = 1.5e3
        x_0 = formulae.trivia.volume(radius=30.531e-6)
        N_0 = 2**23
        sut = Golovin(b)

        # Act
        value = sut.analytic_solution(x=x, t=1200, x_0=x_0, N_0=N_0)

        # Assert
        assert np.all(np.isfinite(value))

    @staticmethod
    @pytest.mark.parametrize("C", (0.0, 1.0))
    def test_simple_geometric(C):
        # arrange
        volume = np.asarray([44.0, 666.0])

        env = Box(dv=None, dt=None)
        builder = Builder(backend=CPU(), n_sd=volume.size, environment=env)
        sut = SimpleGeometric(C=C)
        sut.register(builder)
        _ = builder.build(
            attributes={"volume": volume, "multiplicity": np.ones_like(volume)}
        )

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        output = _PairwiseStorage.from_ndarray(np.zeros_like(volume))
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )
        # act
        sut(output, is_first_in_pair=is_first_in_pair)

        # assert
        if C > 0.0:
            np.testing.assert_array_less([0.0, 0.0], output.to_ndarray())
        else:
            np.testing.assert_array_equal([0.0, 0.0], output.to_ndarray())

    @staticmethod
    @pytest.mark.parametrize("volume", (np.array([1.0, 2.0]), np.array([1.0, 1.0])))
    def test_simple_geometric_same_size(volume):
        # arrange
        env = Box(dv=None, dt=None)
        builder = Builder(backend=CPU(), n_sd=volume.size, environment=env)
        sut = SimpleGeometric(C=1.0)
        sut.register(builder)
        _ = builder.build(
            attributes={"volume": volume, "multiplicity": np.ones_like(volume)}
        )

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        output = _PairwiseStorage.from_ndarray(np.zeros_like(volume))
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )

        # act
        sut(output, is_first_in_pair=is_first_in_pair)

        # assert
        if volume[0] == volume[1]:
            np.testing.assert_array_equal([0.0, 0.0], output.to_ndarray())
        else:
            np.testing.assert_array_less([0.0, 0.0], output.to_ndarray())
