"""
Created at 28.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.particles_builder import ParticlesBuilder


class _Moist:

    def __init__(self, particles_builder: ParticlesBuilder, dt, mesh, variables):
        variables += ['qv', 'thd', 'T', 'p', 'RH']
        self.particles = particles_builder.particles
        self.dt = dt
        self.mesh = mesh
        self._values = {
            "predicted": None,
            "current": self._allocate(variables)
        }
        self._tmp = self._allocate(variables)

    def _allocate(self, variables):
        result = {}
        for var in variables:
            result[var] = self.particles.backend.array((self.mesh.n_cell,), float)
        return result

    def __getitem__(self, index):
        return self._values['current'][index]

    def get_predicted(self, index):
        if self._values['predicted'] is None:
            raise AssertionError("Condensation not called.")
        return self._values['predicted'][index]

    def sync(self):
        target = self._tmp
        self.particles.backend.upload(self._get_qv().ravel(), target['qv'])
        self.particles.backend.upload(self._get_thd().ravel(), target['thd'])

        self.particles.backend.apply(
            function=self.particles.backend.temperature_pressure_RH,
            args=(target['rhod'], target['thd'], target['qv']),
            output=(target['T'], target['p'], target['RH'])
        )
        self._values["predicted"] = target

    def _get_qv(self) -> np.ndarray: raise NotImplemented()
    def _get_thd(self) -> np.ndarray: raise NotImplemented()

    def post_step(self):
        if self._values["predicted"] is None:
            return

        self._tmp = self._values["current"]
        self._values["current"] = self._values["predicted"]
        self._values["predicted"] = None
