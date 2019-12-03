"""
Created at 25.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.simulation.particles import Particles
from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics import constants as const
from PySDM.simulation.environment._moist_air_environment import _MoistAirEnvironment


class AdiabaticParcel(_MoistAirEnvironment):

    def __init__(self, particles: Particles,
                 mass_of_dry_air: float, p0: float, q0: float, T0: float, w: callable, z0: float = 0):

        self.parcel_vars = ['rhod', 'z', 't']
        super().__init__(particles, self.parcel_vars)

        self.m_d = mass_of_dry_air
        self.w = w

        pd0 = p0 * (1 - (1 + const.eps / q0)**-1)

        self['qv'][:] = q0
        self['thd'][:] = phys.th_std(pd0, T0)
        self['rhod'][:] = pd0 / const.Rd / T0
        self['z'][:] = z0
        self['t'][:] = 0

        self.sync_parcel_vars()
        super().sync()
        self.post_step()

    def _get_thd(self):
        return self['thd']

    def _get_qv(self):
        return self['qv']

    # TODO: probably not working outside of running simulation
    # TODO: move back to mesh ! # TODO: better expose m_d (and equip other environments with m_d)
    @property
    def dv(self):
        return self.m_d / self.get_predicted("rhod")[0]

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
