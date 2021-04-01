"""
Created at 2019
"""

from PySDM.state.mesh import Mesh
from PySDM.builder import Builder


class Box:

    def __init__(self, dt, dv):
        self.dt = dt
        self.mesh = Mesh.mesh_0d(dv)
        self.core = None

    def register(self, builder: Builder):
        self.core = builder.core

    def init_attributes(self, *, spectral_discretisation):
        attributes = {}
        attributes['volume'], attributes['n'] = spectral_discretisation.sample(self.core.n_sd)
        return attributes
