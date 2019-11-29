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

        pv0 = p0 / (1 + const.eps / q0)
        pd0 = p0 - pv0
        rhod0 = pd0 / const.Rd / T0
        thd0 = phys.th_std(pd0, T0)
        z0 = 0

        self.t = 0.
        self['qv'][:] = q0
        self['thd'][:] = thd0

        self._tmp['rhod'][:] = rhod0
        self._tmp['z'][:] = z0
        super().sync()

        self['rhod'][:] = rhod0
        self['z'][:] = z0
        self.post_step()

        np.testing.assert_approx_equal(self['T'][:], T0)
        np.testing.assert_approx_equal(self['RH'][:], pv0/phys.pvs(T0))
        np.testing.assert_approx_equal(self['p'][:], p0)
        np.testing.assert_approx_equal(self['qv'][:], q0)
        np.testing.assert_approx_equal(self['rhod'][:], rhod0)
        np.testing.assert_approx_equal(self['thd'][:], thd0)

    def sync(self):
        self.advance_parcel_vars()
        super().sync()

    def advance_parcel_vars(self):
        dt = self.particles.dt

        # mid-point value for w (TODO?)
        self.t += dt
        dz_dt = self.w(self.t - dt/2)

        # Explicit Euler for p,T (predictor step assuming dq=0)
        qv = self['qv'][0]
        T = self['T'][0]
        p = self['p'][0]

        rho = p / phys.R(qv) / T
        pd = p * (1 - 1/ (1 + const.eps / qv))

        dp_dt = - rho * const.g * dz_dt
        dpd_dt = dp_dt
        dT_dt = dp_dt / rho / phys.c_p(qv)

        self._tmp['rhod'][:] += dt * (
                dpd_dt / const.Rd / T +
                -dT_dt * pd / const.Rd / T**2
        )
        self._tmp['z'][:] += dt * dz_dt
        print(self.t, dz_dt, self._tmp['z'][0])



