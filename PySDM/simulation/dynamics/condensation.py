"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State


class Condensation:
    def __init__(self, ambient_air):
        self.ambient_air = ambient_air

    def __call__(self, state: State):

        self.ambient_air.sync()

        # TODO: update drop radii
        #       update fields due to condensation/evaporation
        #       ensure the above does include droplets that precipitated out of the domain



