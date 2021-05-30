import numpy as np


class Mesh:

    def __init__(self, grid, size):
        self.grid = grid
        self.size = size
        self.strides = Mesh.__strides(grid)
        self.n_cell = int(np.prod(grid))
        self.dv = np.prod((np.array(size) / np.array(grid)))
        self.__dimension = len(self.grid)

    @property
    def dz(self):
        return self.size[-1] / self.grid[-1]

    @property
    def dimension(self):
        return self.__dimension

    @property
    def dim(self):
        return self.dimension

    @staticmethod
    def mesh_0d(dv=None):
        mesh = Mesh((1,), ())
        mesh.dv = dv
        mesh.__dimension = 0
        return mesh

    @staticmethod
    def __strides(grid):
        domain = np.empty(tuple(grid))
        strides = np.array(domain.strides) // domain.itemsize
        if len(grid) == 1:
            strides = strides.reshape((1, 1))
        elif len(grid) == 2:
            strides = strides.reshape(1, -1)
        return strides

    def cellular_attributes(self, positions):
        n = positions.shape[1]
        cell_origin = positions.astype(dtype=np.int64)
        position_in_cell = positions - np.floor(positions)

        cell_id = np.empty(n, dtype=np.int64)
        cell_id[:] = np.dot(self.strides, cell_origin)

        return cell_id, cell_origin, position_in_cell
