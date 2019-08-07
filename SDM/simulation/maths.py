"""
Created at 05.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class Maths:
    @staticmethod
    # TODO update
    def moment(state, k, attr='x', attr_range=(0, np.inf)):
        state.housekeeping()

        idx = np.where(np.logical_and(attr_range[0] <= state[attr], state[attr] < attr_range[1]))
        if not idx[0].any():
            return 0 if k == 0 else np.nan
        avg, sum = np.average(state[attr][idx] ** k, weights=state['n'][idx], returned=True)
        return avg * sum