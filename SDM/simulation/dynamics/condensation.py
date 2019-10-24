"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.simulation.state import State


class Condensation:
    def __init__(self, th, qv):
        self.th = th
        self.qv = qv

    def __call__(self, state: State):
        for i in range(state.SD_num):
            th = self.th[state.cell_origin[i]]