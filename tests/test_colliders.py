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
    def __init__(self, returned_value=-1):
        self.returned_value = returned_value

    def __call__(self, m1, m2):
        return self.returned_value


class TestSDM:

    @pytest.mark.parametrize("x, n", [
        pytest.param(np.array([1, 1]), np.array([1, 1])),
        pytest.param(np.array([1, 1]), np.array([5, 1])),
        pytest.param(np.array([1, 1]), np.array([5, 3])),
        pytest.param(np.array([4, 2]), np.array([1, 1])),
    ])
    def test_single_collision(self, x, n):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=0)
        sut.probability = lambda m1, m2, n_sd: 1
        state = State(x, n)

        # Act
        sut(state)

        # Assert
        assert np.sum(state.n * state.x) == np.sum(n * x)
        assert np.sum(state.n) == np.sum(n) - np.amin(n)
        if np.amin(n) > 0: assert np.amax(state.x) == np.sum(x)
        assert np.amax(state.n) == max(np.amax(n) - np.amin(n), np.amin(n))

    @pytest.mark.parametrize("x, n, p", [
        pytest.param(np.array([1, 1]), np.array([1, 1]), 2),
        pytest.param(np.array([1, 1]), np.array([5, 1]), 4),
        pytest.param(np.array([1, 1]), np.array([5, 3]), 5),
        pytest.param(np.array([4, 2]), np.array([1, 1]), 7),
    ])
    def test_multi_collision(self, x, n, p):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=0)
        sut.probability = lambda m1, m2, n_sd: p
        state = State(x, n)

        # Act
        sut(state)

        # Assert
        gamma = min(p, max(n[0] // n[1], n[1] // n[1]))
        assert np.amin(state.n) >= 0
        assert np.sum(state.n * state.x) == np.sum(n * x)
        assert np.sum(state.n) == np.sum(n) - gamma * np.amin(n)
        assert np.amax(state.x) == gamma * x[np.argmax(n)] + x[np.argmax(n) - 1]
        assert np.amax(state.n) == max(np.amax(n) - gamma * np.amin(n), np.amin(n))

    @pytest.mark.parametrize("x, n, p", [
        pytest.param(np.array([1, 1, 1]), np.array([1, 1, 1]), 2),
        pytest.param(np.array([1, 1, 1, 1, 1]), np.array([5, 1, 2, 1, 1]), 1),
        pytest.param(np.array([1, 1, 1, 1, 1]), np.array([5, 1, 2, 1, 1]), 6),
    ])
    def test_multi_droplet(self, x, n, p):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=0)
        sut.probability = lambda x1, x2, n_sd: p
        state = State(x, n)

        # Act
        sut(state)

        # Assert
        assert np.amin(state.n) >= 0
        assert np.sum(state.n * state.x) == np.sum(n * x)

    # TODO integration test?
    def test_multi_step(self):
        # Arrange
        n = np.random.randint(1, 64, size=256)
        x = np.random.uniform(size=256)

        sut = SDM(StubKernel(), dt=0, dv=0)
        sut.probability = lambda x1, x2, n_sd: 0.5
        state = State(x, n)

        from SDM.undertakers import Resize
        undertaker = Resize()

        # Act
        for _ in range(32):
            sut(state)
            undertaker(state)

        # Assert
        assert np.amin(state.n) >= 0
        actual = np.sum(state.n * state.x)
        desired = np.sum(n * x)
        np.testing.assert_almost_equal(actual=actual, desired=desired)

    def test_probability(self):
        # Arrange
        kernel_value = 44
        dt = 666
        dv = 9
        n_sd = 64
        sut = SDM(StubKernel(kernel_value), dt, dv)

        # Act
        actual = sut.probability((0, 1), (0, 1), n_sd)  # TODO dependency state []

        # Assert
        desired = dt/dv * kernel_value * n_sd * (n_sd - 1) / 2 / (n_sd//2)
        assert actual == desired
