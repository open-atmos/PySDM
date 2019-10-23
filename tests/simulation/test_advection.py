"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from SDM.backends.default import Default
from SDM.simulation.advection import Advection
from SDM.simulation.state import State
import numpy as np


class TestAdvection:

    def test_single_cell(self):
        n = np.ones(1)
        n_sd = len(n)
        segments = Default.from_ndarray(np.array([[0, 0]]))
        positions = Default.from_ndarray(np.array([[0.5, 0.5]]))
        courant_field = (np.array([[.1, .2]]).T, np.array([[.3, .4]]))
        state = State(n=n, intensive={}, extensive={}, segments=segments, backend=Default, positions=positions)
        sut = Advection(n_sd=n_sd, courant_field=courant_field, backend=Default)

        sut(state=state)

    def test_advection(self):
        n = np.ones(1)
        n_sd = len(n)
        segments = Default.from_ndarray(np.array([[1, 1]]))
        positions = Default.from_ndarray(np.array([[0.5, 0.5]]))
        courant_field = (np.ones((4, 3)), np.zeros((3, 4)))
        state = State(n=n, intensive={}, extensive={}, segments=segments, backend=Default, positions=positions)
        sut = Advection(n_sd=n_sd, courant_field=courant_field, backend=Default)

        sut(state=state)

        np.testing.assert_array_equal(state.segments[0, :], np.array([2, 1]))

