"""
Created at 25.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics import constants as const
import numpy as np
from PySDM.simulation.environment.moist_air import MoistAir


class AdiabaticParcel(MoistAir):

    def __init__(self, particles, mass, p0, q0, T0, w):

        parcel_vars = ['rhod', 'z']
        super().__init__(particles, parcel_vars)

        self.qv_lambda = lambda: self['qv']
        self.thd_lambda = lambda: self['thd']

        self.mass = mass  # TODO: would be needed for dv (but let's remember it's the total mass - not dry-air mass)
        self.w = w

        pd0 = p0  # TODO !

        self.t = 0.
        self['qv'][:] = q0
        self['thd'][:] = phys.th_std(pd0, T0)

        self._tmp['rhod'][:] = pd0 / const.Rd / T0
        super().sync()

        self.post_step()
        self['z'][:] = 0

        np.testing.assert_approx_equal(self['T'][0], T0)
        # TODO: same for p (after fixing the above pd0 issue !)

    def sync(self):
        self.advance_parcel_vars()
        super().sync()

    def advance_parcel_vars(self):
        dt = self.particles.dt

        # mid-point value for w (TODO?)
        self.t += self.particles.dt
        dz_dt = self.w(self.t - dt/2)

        pv = 0  # TODO !!!!!!!!!!!

        # Explicit Euler for p,T (predictor step)
        dpd_dt = - self['rhod'] * const.g * dz_dt
        dT_dt = dpd_dt / self['rhod'] / phys.c_p(self['qv'][0])  # TODO: consider true dT_dt(p, ...)
        self._tmp['rhod'][:] = self['rhod'] + dt * (
                dpd_dt / const.Rd / self['T'] +
                -dT_dt * (self['p'] - pv) / const.Rd / self['T']**2
        )
        self._tmp['z'][:] = self['z'] + dt * dz_dt



