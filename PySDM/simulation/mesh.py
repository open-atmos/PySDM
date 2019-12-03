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
        self.n_cell = int(np.prod(grid))
        self.dv = np.prod((np.array(size) / np.array(grid)))

    @property
    def dimension(self):
        return len(self.grid)

    @property
    def dim(self):
        return self.dimension

    @staticmethod
    def mesh_0d(dv=None):
        mesh = Mesh((), ())
        mesh.dv = dv
        return mesh

    @staticmethod
    def strides(grid):
        domain = np.empty(tuple(grid))  # TODO optimize
        strides = np.array(domain.strides).reshape(1, -1) // domain.itemsize
        return strides
