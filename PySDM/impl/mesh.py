"""
spatial mesh representation (incl. memory strides)
"""
import numpy as np


class Mesh:
    def __init__(self, grid, size):
        self.grid = grid
        self.size = size
        self.strides = Mesh.__strides(grid)
        self.n_cell = int(np.prod(grid))
        self.dv = np.prod((np.array(size) / np.array(grid)))

    @property
    def dz(self):
        return self.size[-1] / self.grid[-1]

    @property
    def dimension(self):
        return 0 if self.grid == (1,) else len(self.grid)

    @property
    def dim(self):
        return self.dimension

    @staticmethod
    def mesh_0d(dv=None):
        mesh = Mesh((1,), ())
        mesh.dv = dv
        return mesh

    @staticmethod
    def __strides(grid):
        """
        returns strides, i.e.: to compute a `cell_id`, use np.dot(strides, cell_origin)

        returns the stride vector for a given grid (for use in `cell_id` arithmetics where
        the stride vector indicates the distances of cell ids adjacent in a given dimension)
        """
        domain = np.empty(tuple(grid))
        strides = np.array(domain.strides) // domain.itemsize
        if len(grid) == 1:
            strides = strides.reshape((1, 1))
        else:
            strides = strides.reshape(1, -1)
        return strides

    def cellular_attributes(self, positions):
        """
        computes values of `cell_id`, `cell_origin` and `position_in_cell` attributes
        based on `positions` values passed as input

        takes:
          droplet `positions` as input (expressed in grid coordinates, i.e.
          in a range of 0 ... 10, 0 ... 5 for a 10x5 grid)

        returns:
         a tuple of:
           - `cell_id`: scalar integer ids of cells
             (each value within the range of 0...prod(grid))
           - `cell_origin`: n-component vector of integer cell coordinates
             (each within the range of 0...grid[dim])
           - `position_in_cell`: n-component vector of floats
             (each within the range of 0...1)
        """
        n = positions.shape[1]
        cell_origin = positions.astype(dtype=np.int64)
        position_in_cell = positions - np.floor(positions)

        cell_id = np.empty(n, dtype=np.int64)
        cell_id[:] = np.dot(self.strides, cell_origin)

        return cell_id, cell_origin, position_in_cell
