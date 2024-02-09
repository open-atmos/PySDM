"""
Bare zero-dimensional framework
"""

import numpy as np

from PySDM.impl.mesh import Mesh


class Box:
    def __init__(self, dt, dv):
        self.dt = dt
        self.mesh = Mesh.mesh_0d(dv)
        self.particulator = None
        self._ambient_air = {}

    def __getitem__(self, item):
        return self._ambient_air[item]

    def __setitem__(self, key, value):
        if key not in self._ambient_air:
            self._ambient_air[key] = self.particulator.backend.Storage.from_ndarray(
                np.array([value])
            )
        else:
            self._ambient_air[key][:] = value

    def register(self, builder):
        self.particulator = builder.particulator

    def init_attributes(self, *, spectral_discretisation):
        attributes = {}
        (
            attributes["volume"],
            attributes["multiplicity"],
        ) = spectral_discretisation.sample(
            backend=self.particulator.backend, n_sd=self.particulator.n_sd
        )
        return attributes
