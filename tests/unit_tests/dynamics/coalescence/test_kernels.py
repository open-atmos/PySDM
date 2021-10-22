import numpy as np
import pytest
from PySDM.physics.formulae import Formulae
from PySDM.physics.coalescence_kernels import Golovin


class TestGolovin:

    @staticmethod
    @pytest.mark.parametrize("x", [
        pytest.param(5e-10),
        pytest.param(np.full(10, 5e-10))
    ])
    def test_analytic_solution_underflow(x):
        # Arrange
        formulae = Formulae()
        b = 1.5e3
        x_0 = formulae.trivia.volume(radius=30.531e-6)
        N_0 = 2 ** 23
        sut = Golovin(b)

        # Act
        value = sut.analytic_solution(x=x, t=1200, x_0=x_0, N_0=N_0)

        # Assert
        assert np.all(np.isfinite(value))
