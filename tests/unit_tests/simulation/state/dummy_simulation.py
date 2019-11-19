"""
Created at 19.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from tests.unit_tests.simulation.state.testable_state_factory import TestableStateFactory


class DummySimulation:

    def __init__(self, backend):

        self.backend = backend

    def add_attrs(self, dt=None, dv=None, n_sd=None, n_cell=None, grid=None):
        self.dt = dt
        self.dv = dv
        self.n_sd = n_sd
        self.n_cell = n_cell
        self.grid = grid
