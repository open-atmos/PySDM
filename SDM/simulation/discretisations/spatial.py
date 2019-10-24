"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


# http://ww2.ii.uj.edu.pl/~arabas/workshop_2019/files/talk_Shima.pdf


def grid(grid, n_sd):
    raise NotImplementedError()


def pseudorandom(grid, n_sd):
    dimension = len(grid)
    positions = np.random.rand(n_sd, dimension)
    for i in range(dimension):
        positions[:, i] *= grid[i]

    return positions


def quasirandom(grid, n_sd):
    raise NotImplementedError()
