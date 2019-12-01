"""
Created at 28.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class Mesh:
    def __init__(self, grid, size):
        self.grid = grid
        self.size = size
        self.strides = Mesh.strides(grid)
        self.n_cell = grid[0] * grid[1]
        self.dv = (size[0] / grid[0]) * (size[1] / grid[1])

    @property
    def dimension(self):
        return len(self.grid)

    @property
    def dim(self):
        return self.dimension

    # TODO hardcoded 2D in constructor
    @staticmethod
    def mesh_0d(dv=None):
        mesh = Mesh((1, 1), (1, 1))
        mesh.grid = ()
        mesh.size = ()
        mesh.strides = None
        mesh.n_cell = 1
        mesh.dv = dv
        return mesh

    @staticmethod
    def strides(grid):
        domain = np.empty(tuple(grid))  # TODO optimize
        strides = np.array(domain.strides).reshape(1, -1) // domain.itemsize
        return strides
