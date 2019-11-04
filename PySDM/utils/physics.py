"""
Created at 05.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class Physics:
    @staticmethod
    def x2r(x):
        return (x * 3 / 4 / np.pi) ** (1 / 3)

    @staticmethod
    def r2x(r):
        return 4 / 3 * np.pi * r ** 3

    kg2g = 1e3
    m2um = 1e6
