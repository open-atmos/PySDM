"""
Created at 06.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.colliders import SDM
from SDM.state import State
import numpy as np
import pytest


class StubKernel:
    def __init__(self, retval=-1):
        self.retval = retval

    def __call__(self, m1, m2):
        return self.retval


class TestSDM:

    @pytest.mark.parametrize("m, n", [
        pytest.param(np.array([1, 1]), np.array([1, 1])),
        pytest.param(np.array([1, 1]), np.array([5, 1])),
        pytest.param(np.array([1, 1]), np.array([5, 3])),
        pytest.param(np.array([4, 2]), np.array([1, 1])),
    ])
    def test_single_collision(self, m, n):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=0)
        sut.probability = lambda m1, m2, n_sd: 1
        state = State(m, n)

        # Act
        sut(state)

        # Assert
        assert np.sum(state.n * state.m) == np.sum(n * m)
        assert np.sum(state.n) == np.sum(n) - np.amin(n)
        if np.amin(n) > 0: assert np.amax(state.m) == np.sum(m)
        assert np.amax(state.n) == max(np.amax(n) - np.amin(n), np.amin(n))

    @pytest.mark.parametrize("m, n, p", [
        pytest.param(np.array([1, 1]), np.array([1, 1]), 2),
        pytest.param(np.array([1, 1]), np.array([5, 1]), 4),
        pytest.param(np.array([1, 1]), np.array([5, 3]), 5),
        pytest.param(np.array([4, 2]), np.array([1, 1]), 7),
    ])
    def test_multi_collision(self, m, n, p):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=0)
        sut.probability = lambda m1, m2, n_sd: p
        state = State(m, n)

        # Act
        sut(state)

        # Assert
        gamma = min(p, max(n[0] // n[1], n[1] // n[1]))
        assert np.amin(state.n) >= 0
        assert np.sum(state.n * state.m) == np.sum(n * m)
        assert np.sum(state.n) == np.sum(n) - gamma * np.amin(n)
        if np.amin(n) > 0: assert np.amax(state.m) == gamma * m[np.argmax(n)] + m[np.argmax(n) - 1]
        assert np.amax(state.n) == max(np.amax(n) - gamma * np.amin(n), np.amin(n))

    def test_multi_droplet(self):
        pass # TODO

    def test_probability(self):
        # Arrange
        kerval = 44
        dt = 666
        dv = 9
        n_sd = 64
        sut = SDM(StubKernel(kerval), dt, dv)

        # Act
        actual = sut.probability(0, 0, n_sd)

        # Assert
        desired = dt/dv * kerval * n_sd * (n_sd - 1) / 2 / (n_sd//2)
        assert actual == desired
