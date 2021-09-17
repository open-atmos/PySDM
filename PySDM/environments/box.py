"""
Bare zero-dimensional framework
"""

from PySDM.state.mesh import Mesh


class Box:

    def __init__(self, dt, dv):
        self.dt = dt
        self.mesh = Mesh.mesh_0d(dv)
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

    def init_attributes(self, *, spectral_discretisation):
        attributes = {}
        attributes['volume'], attributes['n'] = spectral_discretisation.sample(self.particulator.n_sd)
        return attributes
