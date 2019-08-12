"""
Created at 05.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class Maths:
    @staticmethod
    def moment(state, k, attr='x', attr_range=(0, np.inf)):
        idx = np.where(np.logical_and(attr_range[0] <= state[attr], state[attr] < attr_range[1]))
        if not idx[0].any():
            return 0 if k == 0 else np.nan
        avg, sum_of_n = np.average(state[attr][idx] ** k, weights=state['n'][idx], returned=True)
        return avg * sum_of_n

    @staticmethod
    def test():
        print("magic")
