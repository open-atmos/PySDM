"""
Created at 2019
"""

from PySDM.mesh import Mesh
from PySDM.builder import Builder


class Box:

    def __init__(self, dt, dv=None):
        self.dt = dt
        self.mesh = Mesh.mesh_0d(dv)

    def register(self, _: Builder):
        pass
