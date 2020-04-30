"""
Created at 25.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.particles import Particles
from PySDM.physics import formulae as phys
from ..physics import constants as const
from ._moist import _Moist
from ._moist_lagrangian_parcel import _MoistLagrangianParcel
from PySDM.mesh import Mesh


class MoistLagrangianParcelAdiabatic(_MoistLagrangianParcel):

    def __init__(self, particles: Particles, dt,
                 mass_of_dry_air: float, p0: float, q0: float, T0: float, w: callable, z0: float = 0):

        super().__init__(particles, dt, Mesh.mesh_0d(), ['rhod', 'z', 't'], mass_of_dry_air)

        # TODO: move w-related logic to _MoistLagrangianParcel
        self.w = w

        pd0 = p0 * (1 - (1 + const.eps / q0)**-1)

        self['qv'][:] = q0
        self['thd'][:] = phys.th_std(pd0, T0)
        self['rhod'][:] = pd0 / const.Rd / T0
        self['z'][:] = z0
        self['t'][:] = 0

        self.sync_parcel_vars()
        _Moist.sync(self)
        self.post_step()

    def advance_parcel_vars(self):
        dt = self.particles.dt
        qv = self['qv'][0]
        T = self['T'][0]
        p = self['p'][0]
        t = self['t'][0]

        rho = p / phys.R(qv) / T
        pd = p * (1 - 1 / (1 + const.eps / qv))

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
