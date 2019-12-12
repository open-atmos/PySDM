"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.dynamics.coalescence.kernels.golovin import Golovin
import numpy as np
import pytest


class TestGolovin:

    @pytest.mark.parametrize("x", [
        pytest.param(5e-10),
        pytest.param(np.full(10, 5e-10))
    ])
    def test_analytic_solution_underflow(self, x):
        # Arrange
        b = 1.5e3
        x_0 = 4 / 3 * np.pi * 30.531e-6 ** 3
        N_0 = 2 ** 23
        sut = Golovin(b)

        # Act
        value = sut.analytic_solution(x=x, t=1200, x_0=x_0, N_0=N_0)

        # Assert
        assert np.all(np.isfinite(value))
