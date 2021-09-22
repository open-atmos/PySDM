"""
Bare zero-dimensional framework
"""

from PySDM.state.mesh import Mesh
import numpy as np


class Box:

    def __init__(self, dt, dv):
        self.dt = dt
        self.mesh = Mesh.mesh_0d(dv)
        self.particulator = None
        self._ambient_air = {}

    def __getitem__(self, item):
        return self._ambient_air[item]

    def __setitem__(self, key, value):
        self._ambient_air[key] = self.particulator.backend.Storage.from_ndarray(np.array([value]))

    def register(self, builder):
        self.particulator = builder.particulator

    def init_attributes(self, *, spectral_discretisation):
        attributes = {}
        attributes['volume'], attributes['n'] = spectral_discretisation.sample(self.particulator.n_sd)
        return attributes
