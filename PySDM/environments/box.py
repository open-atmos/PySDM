"""
Bare zero-dimensional framework
"""

from PySDM.state.mesh import Mesh


class Box:

    def __init__(self, dt, dv):
        self.dt = dt
        self.mesh = Mesh.mesh_0d(dv)
        self.core = None

    def register(self, builder):
        self.core = builder.core

    def init_attributes(self, *, spectral_discretisation):
        attributes = {}
        attributes['volume'], attributes['n'] = spectral_discretisation.sample(self.core.n_sd)
        return attributes
