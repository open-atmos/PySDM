"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Domain:
    def __init__(self, dimension, grid, size):
        # TODO assert
        self.dimension = dimension
        self.grid = grid
        self.size = size
        self.grid_step = [self.size[dim] / self.grid[dim] for dim in range(dimension)]