"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.undertakers import Resize
from SDM.state import State
import pytest
import numpy as np


class TestResize:

    @pytest.mark.parametrize("x, n", [
        pytest.param(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])),
        pytest.param(np.array([1, 2, 1, 1]), np.array([2, 0, 2, 0])),
        pytest.param(np.array([1, 1, 4]), np.array([5, 0, 0]))
    ])
    def test___call__(self, x, n):
        sut = Resize()
        state = State(x, n)

        sut(state)

        assert state.x.shape == state.n.shape
        assert state.n.shape[0] == (n != 0).sum()
        assert state.n.sum() == n.sum()
        assert (state.x * state.n).sum() == (x * n).sum()
