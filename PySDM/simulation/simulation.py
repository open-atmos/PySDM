"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State
from PySDM.simulation.state.state_factory import StateFactory


# TODO: better name?
class Simulation:

    def __init__(self, backend):

        self.backend = backend()
        self.state: State = None
        self.dynamics: list = []

    def add_dynamics(self):
        if self.state is None:  # TODO need n, grid etc.
            raise AssertionError("State is None.")

    def create_state_0d(self, n, intensive, extensive):
        self.state = StateFactory.state_0d(n, intensive, extensive, self)

