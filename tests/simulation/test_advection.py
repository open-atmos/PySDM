"""
Created at 23.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from SDM.backends.default import Default
from SDM.simulation.dynamics.advection import Advection
from SDM.simulation.state import State
import numpy as np


class TestAdvection:

    def test_single_cell(self):
        n = np.ones(1)
        n_sd = len(n)
        positions = Default.from_ndarray(np.array([[0.5, 0.5]]))
        courant_field = (np.array([[.1, .2]]).T, np.array([[.3, .4]]))
        state = State.state_2d(n=n, grid=(1, 1), intensive={}, extensive={}, positions=positions, backend=Default)
        sut = Advection(n_sd=n_sd, courant_field=courant_field, backend=Default)

        sut(state=state)

    def test_advection(self):
        n = np.ones(1)
        n_sd = len(n)
        positions = Default.from_ndarray(np.array([[1.5, 1.5]]))
        courant_field = (np.ones((4, 3)), np.zeros((3, 4)))
        state = State.state_2d(n=n, grid=(3, 3), intensive={}, extensive={}, backend=Default, positions=positions)
        sut = Advection(n_sd=n_sd, courant_field=courant_field, backend=Default)

        sut(state=state)

        np.testing.assert_array_equal(state.cell_origin[0, :], np.array([2, 1]))

