

from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics import constants as const
import numpy as np


class AdiabaticParcel: # TODO: inherit from environmrnt.moist_air!

    def __init__(self, particles, mass, p, q, T, w):
        self.mass = mass
        self.pd = p # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.q = q
        self.T = T
        self.w = w
        self.particles = particles

        self.t = 0.

        self._values = {
            "new": None,
            "old": self._allocate()
        }
        self._tmp = self._allocate()

        # TODO
        self.sync()
        self.post_step()

    @property
    def rhod(self):
        return self.pd / const.Rd / self.T

    @property
    def dv(self):
        raise NotImplementedError()

    @property
    def n_cell(self):
        return 1

    def __getitem__(self, index):
        values = self._values[index]
        if values is None:
            raise Exception("condensation not called.")
        return values

    def _allocate(self):
        result = {}
        for var in ['qv', 'thd']:
            result[var] = self.particles.backend.array((self.n_cell,), float)
        return result

    def sync(self):
        qv = np.full((1,), self.q)
        thd = np.full((1,), phys.th_std(self.pd, self.T))

        target = self._tmp
        self.particles.backend.upload(qv, target['qv'])
        self.particles.backend.upload(thd, target['thd'])

    def ante_step(self):
        dt = self.particles.dt
        # Explicit Euler for p,T; with mid-point w
        w = self.w(self.t + dt/2)
        dpd_dt = - self.rhod() * const.g * w
        dT_dt = dpd_dt / self.rhod / phys.c_p(self.q) # TODO: consider true dT_dt(p, ...)
        self.pd += dpd_dt * dt
        self.T += dT_dt * dt

    def post_step(self):
        backend = self.particles.backend

        qv = np.empty((1,))
        thd = np.empty((1,))
        backend.download(self._values["new"]["qv"], qv)
        backend.download(self._values["new"]["thd"], thd)
        _, T, _ = backend.temperature_pressure_RH(self.rhod, qv[0], thd[0])
        self.q = qv[0]
        self.T = T

        self._tmp = self._values["old"]
        self._values["old"] = self._values["new"]
        self._values["new"] = None
