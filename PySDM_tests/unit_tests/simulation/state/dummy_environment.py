"""
Created at 21.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.mesh import Mesh


class DummyEnvironment:

    def __init__(self, _, grid=None, size=None, dv=None, courant_field_data=None):
        print(grid)
        if grid is None:
            self.mesh = Mesh.mesh_0d(dv)
        else:
            if size is None:
                size = tuple(1 for _ in range(len(grid)))
            self.mesh = Mesh(grid, size)
        self.courant_field_data = courant_field_data

    def get_courant_field_data(self):
        return self.courant_field_data
