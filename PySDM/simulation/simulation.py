"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State


# TODO: better name?
class Simulation:

    def __init__(self, backend):

        self.backend = backend()
        self.state: State = None
        self.dynamics: list = []

    def add_dynamics(self):
        if self.state is None:
            raise AssertionError("State is None.")

    def create_state_0d(self):
        pass

