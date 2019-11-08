"""
Created at 05.08.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.simulation.state import State


class Maths:
    @staticmethod
    def moment_0d(state, k, attr='x', attr_range=(0, np.inf), cell_id=0):
        idx = np.where(np.logical_and(
            np.logical_and(attr_range[0] <= state[attr], state[attr] < attr_range[1]),
            cell_id == state['cell_id']  # TODO
        ))
        if not idx[0].any():
            return 0 if k == 0 else np.nan
        avg, sum_of_n = np.average(state[attr][idx] ** k, weights=state['n'][idx], returned=True)
        return avg * sum_of_n

