"""
Created at 19.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.mesh import Mesh


class DummyParticles:

    def __init__(self, backend, n_sd, dt=None):
        self.backend = backend
        self.n_sd = n_sd
        self.dt = dt
        self.mesh = None
        self.environment = None

    def set_mesh(self, grid, size=None):
        if size is None:
            size = tuple(1 for _ in range(len(grid)))
        self.mesh = Mesh(grid, size)

    def set_mesh_0d(self, dv=None):
        self.mesh = Mesh.mesh_0d(dv)

    def set_environment(self, environment_class, params):
        self.environment = environment_class(None, *params)
