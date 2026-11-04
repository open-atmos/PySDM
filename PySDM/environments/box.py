"""
Bare zero-dimensional framework
"""

import numpy as np

from PySDM.impl.mesh import Mesh
from PySDM.environments.impl import register_environment


@register_environment()
class Box:
    def __init__(self, dt, dv, *, backend=None):
        self.dt = dt
        self.mesh = Mesh.mesh_0d(dv)
        self.particulator = None
        self._ambient_air = {}
        self.backend = backend
        if backend is not None:
            assert backend.n_dims == self.mesh.n_dims

    def __getitem__(self, item):
        return self._ambient_air[item]

    def __setitem__(self, key, value):
        if key not in self._ambient_air:
            self._ambient_air[key] = self.backend.Storage.from_ndarray(
                np.array([value])
            )
        else:
            self._ambient_air[key][:] = value

    def register(self, particulator):
        self.particulator = particulator
