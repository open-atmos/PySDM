"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state import State
import numpy as np


class Condensation:
    def __init__(self, ambient_air):
        self.ambient_air = ambient_air

    def __call__(self, state: State):

        self.ambient_air.sync()

        # update drop radii



