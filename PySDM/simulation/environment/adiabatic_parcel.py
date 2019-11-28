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

        super().__init__(particles, ['qv', 'thd', 'T', 'p', 'RH', 'rhod'])
        self.mass = mass
        self.w = w

        self.t = 0.

        pd0 = p0  # TODO !
        self['qv'][:] = q0
        self['thd'][:] = phys.th_std(pd0, T0)
        self['rhod'][:] = pd0 / const.Rd / T0

        self.qv_lambda = lambda: self['qv']
        self.thd_lambda = lambda: self['thd']

        self.sync()
        self._values['predicted']['rhod'] = self['rhod']

        np.testing.assert_approx_equal(self._values['predicted']['T'][0], T0)
        # TODO: same for p (after fixing the above pd0 issue !!!!!)

        self._update()

    @property
    def rhod(self):
        return self['rhod']

    def ante_step(self):
        dt = self.particles.dt

        # mid-point value for w (TODO?)
        self.t += self.particles.dt
        w = self.w(self.t - dt/2)

        pv = 0 # TODO!!!!!!!

        # Explicit Euler for p,T (predictor step)
        dpd_dt = - self['rhod'] * const.g * w
        dT_dt = dpd_dt / self['rhod'] / phys.c_p(self['qv'][0]) # TODO: consider true dT_dt(p, ...)
        self['rhod'][:] = self['rhod'] + dt * (
                dpd_dt / const.Rd / self['T'] +
                -dT_dt * (self['p'] - pv) / const.Rd / self['T']**2
        )
        pass

    def post_step(self):
        self._update()
