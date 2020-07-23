"""
Created at 24.10.2019
"""

# http://ww2.ii.uj.edu.pl/~arabas/workshop_2019/files/talk_Shima.pdf

import numpy as np


def grid(grid, n_sd):
    raise NotImplementedError()


def pseudorandom(grid, n_sd):
    dimension = len(grid)
    positions = np.random.rand(dimension, n_sd)
    for dim in range(dimension):
        positions[dim, :] *= grid[dim]
    return positions


def quasirandom(grid, n_sd):
    raise NotImplementedError()
