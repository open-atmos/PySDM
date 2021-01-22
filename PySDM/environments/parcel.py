"""
Created at 25.11.2019
"""

import numpy as np
from PySDM.initialisation.r_wet_init import r_wet_init, default_rtol
from PySDM.initialisation.multiplicities import discretise_n
from PySDM.physics import formulae as phys
from ..physics import constants as const
from ._moist import _Moist
from PySDM.state.mesh import Mesh


class Parcel(_Moist):

    def __init__(
            self, dt,
            mass_of_dry_air: float,
            p0: float, q0: float, T0: float,
            w: [float, callable], z0: float = 0):

        super().__init__(dt, Mesh.mesh_0d(), ['rhod', 'z', 't'])

        self.w = w if callable(w) else lambda _: w

        pd0 = p0 * (1 - (1 + const.eps / q0)**-1)
        rhod0 = pd0 / const.Rd / T0

        self.params = (q0, phys.th_std(pd0, T0), rhod0, z0, 0)
        self.mesh.dv = mass_of_dry_air / rhod0

        self.mass_of_dry_air = mass_of_dry_air

    @property
    def dv(self):
        rhod_mean = (self.get_predicted("rhod")[0] + self["rhod"][0]) / 2
        return self.mass_of_dry_air / rhod_mean

    def register(self, builder):
        _Moist.register(self, builder)
        self['qv'][:] = self.params[0]
        self['thd'][:] = self.params[1]
        self['rhod'][:] = self.params[2]
        self['z'][:] = self.params[3]
        self['t'][:] = self.params[4]
        delattr(self, 'params')

        self.sync_parcel_vars()
        _Moist.sync(self)
        self.notify()

    def init_attributes(self, *, n_in_dv: [float, np.ndarray], kappa: float, r_dry: [float, np.ndarray],
                        rtol=default_rtol):
        if not isinstance(n_in_dv, np.ndarray):
            r_dry = np.array([r_dry])
            n_in_dv = np.array([n_in_dv])

        attributes = {}
        attributes['dry volume'] = phys.volume(radius=r_dry)
        attributes['n'] = discretise_n(n_in_dv)
        r_wet = r_wet_init(r_dry, self, np.zeros_like(attributes['n']), kappa, rtol)
        attributes['volume'] = phys.volume(radius=r_wet)
        return attributes

    def advance_parcel_vars(self):
        dt = float(self.core.dt)
        qv = float(self['qv'][0])
        T = float(self['T'][0])
        p = float(self['p'][0])
        t = float(self['t'][0])

        rho = p / phys.R(qv) / T
        pd = p * (1 - 1 / (1 + const.eps / qv))

        # mid-point value for w
        dz_dt = self.w(t + dt/2)
        dz_dt = float(dz_dt)

        # Explicit Euler for p,T (predictor step assuming dq=0)
        dp_dt = - rho * const.g * dz_dt
        dpd_dt = dp_dt  # dq=0
        dT_dt = dp_dt / rho / phys.c_p(qv)

        dt_as_storage = self.core.Storage.from_ndarray(np.array([dt]))

        dt_dzdt =  self.core.Storage.from_ndarray(np.array([dt*dz_dt]))
        rhodRightSide = self.core.Storage.from_ndarray(np.array([(dt * (
                dpd_dt / const.Rd / T +
                -dT_dt * pd / const.Rd / T**2))]))

        self._tmp['t'][:] += dt_as_storage
        self._tmp['z'][:] += dt_dzdt
        self._tmp['rhod'][:] += rhodRightSide

        rhod_mean = (self._tmp['rhod'][0] + self["rhod"][0]) / 2
        self.mesh.dv = self.mass_of_dry_air / rhod_mean

    def get_thd(self):
        return self['thd']

    def get_qv(self):
        return self['qv']

    def sync_parcel_vars(self):
        for var in self.variables:
            self._tmp[var][:] = self[var][:]

    def sync(self):
        self.sync_parcel_vars()
        self.advance_parcel_vars()
        super().sync()
