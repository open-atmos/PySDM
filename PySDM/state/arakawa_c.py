import numpy as np


def z_scalar_coord(grid):
    zZ = np.linspace(1/2, grid[-1]-1/2, grid[-1])
    return zZ


def make_rhod(grid, rhod_of_zZ):
    return np.repeat(
        rhod_of_zZ(
            z_scalar_coord(grid) / grid[-1]
        ).reshape((1, grid[1])),
        grid[0],
        axis=0
    )

