"""
Created at 06.06.2019

@author: Piotr Bartman
"""

from SDM.dynamics import Dynamic
from SDM.state import State
import numpy as np
import pytest

class TestDynamic:

    @pytest.mark.parametrize("m, n", [
        pytest.param(np.array([1, 1]), np.array([1, 1])),
        pytest.param(np.array([1, 1]), np.array([1, 0])),
        pytest.param(np.array([1, 1]), np.array([5, 1])),
        pytest.param(np.array([1, 1]), np.array([5, 3])),
        pytest.param(np.array([1, 1]), np.array([0, 0])),
        pytest.param(np.array([4, 2]), np.array([1, 1])),
    ])
    def test_step(self, m, n):
        # Arrange
        sut = Dynamic(lambda p1, p2: 1)
        state = State(m, n)

        # Act
        sut.step(state)

        # Assert
        assert np.sum(state.n * state.m) == np.sum(n * m)
        assert np.sum(state.n) == np.sum(n) - np.amin(n)
        if np.amin(n) > 0: assert np.amax(state.m) == np.sum(m)
        assert np.amax(state.n) == max(np.amax(n) - np.amin(n), np.amin(n))


    @pytest.mark.parametrize("m, n, p", [
        pytest.param(np.array([1, 1]), np.array([1, 1]), 2),
        pytest.param(np.array([1, 1]), np.array([1, 0]), 3),
        pytest.param(np.array([1, 1]), np.array([5, 1]), 4),
        pytest.param(np.array([1, 1]), np.array([5, 3]), 5),
        pytest.param(np.array([1, 1]), np.array([0, 0]), 6),
        pytest.param(np.array([4, 2]), np.array([1, 1]), 7),
    ])
    def test_step_multicollision(self, m, n, p):
        # Arrange
        sut = Dynamic(lambda p1, p2: p)
        state = State(m, n)

        # Act
        sut.step(state)

        # Assert
        print(state.n, state.m)
        assert np.amin(state.n) >= 0
        assert np.sum(state.n * state.m) == np.sum(n * m)
        assert np.sum(state.n) == np.sum(n) - p * np.amin(n)
        if np.amin(n) > 0: assert np.amax(state.m) == np.sum(m)
        assert np.amax(state.n) == max(np.amax(n) - np.amin(n), np.amin(n))