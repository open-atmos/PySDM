"""
Created at 25.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.simulation.particles import Particles
from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics import constants as const
import numpy as np
from PySDM.simulation.environment._moist_air_environment import _MoistAirEnvironment


class AdiabaticParcel(_MoistAirEnvironment):

    def __init__(self, particles: Particles, mass: float, p0: float, q0: float, T0: float, w: callable):

        self.parcel_vars = ['rhod', 'z', 't']
        super().__init__(particles, self.parcel_vars)

        self.qv_lambda = lambda: self['qv']
        self.thd_lambda = lambda: self['thd']

        self.mass = mass  # TODO: would be needed for dv (but let's remember it's the total mass - not dry-air mass)
        self.w = w

        pv0 = p0 / (1 + const.eps / q0)
        pd0 = p0 - pv0
        rhod0 = pd0 / const.Rd / T0
        thd0 = phys.th_std(pd0, T0)

        self['qv'][:] = q0
        self['thd'][:] = thd0
        self['rhod'][:] = rhod0
        self['z'][:] = 0
        self['t'][:] = 0

        self.sync_parcel_vars()
        super().sync()
        self.post_step()

        np.testing.assert_approx_equal(self['T'][:], T0)
        np.testing.assert_approx_equal(self['RH'][:], pv0/phys.pvs(T0))
        np.testing.assert_approx_equal(self['p'][:], p0)
        np.testing.assert_approx_equal(self['qv'][:], q0)
        np.testing.assert_approx_equal(self['rhod'][:], rhod0)
        np.testing.assert_approx_equal(self['thd'][:], thd0)

    def sync_parcel_vars(self):
        for var in self.parcel_vars:
            self._tmp[var][:] = self[var][:]

    def sync(self):
        self.sync_parcel_vars()
        self.advance_parcel_vars()
        super().sync()

    def advance_parcel_vars(self):
        dt = self.particles.dt
        qv = self['qv'][0]
        T = self['T'][0]
        p = self['p'][0]
        t = self['t'][0]

        rho = p / phys.R(qv) / T
        pd = p * (1 - 1/ (1 + const.eps / qv))

        # mid-point value for w
        dz_dt = self.w(t + dt/2)

        # Explicit Euler for p,T (predictor step assuming dq=0)
        dp_dt = - rho * const.g * dz_dt
        dpd_dt = dp_dt  # dq=0
        dT_dt = dp_dt / rho / phys.c_p(qv)

        self._tmp['t'][:] += dt
        self._tmp['z'][:] += dt * dz_dt
        self._tmp['rhod'][:] += dt * (
                dpd_dt / const.Rd / T +
                -dT_dt * pd / const.Rd / T**2
        )
        # TODO: do RK4 for all the above...
