

from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics import constants as const
import numpy as np


class AdiabaticParcel:

    def __init__(self, particles, mass, p, q, T, w):
        self.mass = mass
        self.p = p
        self.q = q
        self.T = T
        self.w = w
        self.particles = particles

        self._values = {
            "new": None,
            "old": self._allocate()
        }
        self._tmp = self._allocate()

        # TODO
        self.sync()
        self.post_step()

    @property
    def rho(self):
        return self.p / phys.R(self.q) / self.T

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
        thd = np.full((1,), phys.th_dry(phys.th_std(self.p, self.T), self.q))

        target = self._tmp
        self.particles.backend.upload(qv, target['qv'])
        self.particles.backend.upload(thd, target['thd'])

    def ante_step(self):
        # self.T += (self.T * phys.R(self.q) / self.p *)
    #     self.p += - (self.p/ phys.R(self.q) /self.T) * self.w * const.g * dt
    #     # z +=  w * dt
        pass

    def post_step(self):
        qv = np.empty((1,))
        thd = np.empty((1,))
        self.particles.backend.download(self._values["new"]["qv"], qv)
        self.particles.backend.download(self._values["new"]["thd"], thd)
        self.q = qv
        self.T = phys.T_from_thd(thd, self.p)

        self._tmp = self._values["old"]
        self._values["old"] = self._values["new"]
        self._values["new"] = None
