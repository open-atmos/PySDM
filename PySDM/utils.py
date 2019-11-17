"""
Created at 05.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


# TODO: move to example
class Physics:
    @staticmethod
    def x2r(x):
        return (x * 3 / 4 / np.pi) ** (1 / 3)

    @staticmethod
    def r2x(r):
        return 4 / 3 * np.pi * r ** 3


def strides(grid):
    domain = np.empty(tuple(grid))  # TODO optimize
    strides = np.array(domain.strides).reshape(1, -1) // domain.itemsize
    return strides
