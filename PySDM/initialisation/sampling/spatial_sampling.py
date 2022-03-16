"""
spatial sampling logic (i.e., physical x-y-z coordinates)
"""
import numpy as np

# TODO #305 QUASIRANDOM & GRID
#  http://www.ii.uj.edu.pl/~arabas/workshop_2019/files/talk_Shima.pdf


class Pseudorandom:
    @staticmethod
    def sample(grid, n_sd):
        dimension = len(grid)
        positions = np.random.rand(dimension, n_sd)
        for dim in range(dimension):
            positions[dim, :] *= grid[dim]
        return positions
