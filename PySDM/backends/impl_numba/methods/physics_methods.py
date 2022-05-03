"""
CPU implementation of backend methods wrapping basic physics formulae
"""
import numba
from numba import prange

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf


class PhysicsMethods(BackendMethods):
    def __init__(self):
        super().__init__()
        pvs_C = self.formulae.saturation_vapour_pressure.pvs_Celsius
        pvi_C = self.formulae.saturation_vapour_pressure.ice_Celsius
        phys_T = self.formulae.state_variable_triplet.T
        phys_p = self.formulae.state_variable_triplet.p
        phys_pv = self.formulae.state_variable_triplet.pv
        explicit_euler = self.formulae.trivia.explicit_euler
        phys_sigma = self.formulae.surface_tension.sigma
        phys_volume = self.formulae.trivia.volume
        phys_r_cr = self.formulae.hygroscopicity.r_cr
        const = self.formulae.constants

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def explicit_euler_body(y, dt, dy_dt):
            y[:] = explicit_euler(y, dt, dy_dt)

        self.explicit_euler_body = explicit_euler_body

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def critical_volume(*, v_cr, kappa, f_org, v_dry, v_wet, T, cell):
            for i in prange(len(v_cr)):  # pylint: disable=not-an-iterable
                sigma = phys_sigma(T[cell[i]], v_wet[i], v_dry[i], f_org[i])
                v_cr[i] = phys_volume(
                    phys_r_cr(
                        kp=kappa[i],
                        rd3=v_dry[i] / const.PI_4_3,
                        T=T[cell[i]],
                        sgm=sigma,
                    )
                )

        self.critical_volume_body = critical_volume

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def temperature_pressure_RH_body(*, rhod, thd, qv, T, p, RH):
            for i in prange(T.shape[0]):  # pylint: disable=not-an-iterable
                T[i] = phys_T(rhod[i], thd[i])
                p[i] = phys_p(rhod[i], T[i], qv[i])
                RH[i] = phys_pv(p[i], qv[i]) / pvs_C(T[i] - const.T0)

        self.temperature_pressure_RH_body = temperature_pressure_RH_body

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def terminal_velocity_body(*, values, radius, k1, k2, k3, r1, r2):
            for i in prange(len(values)):  # pylint: disable=not-an-iterable
                if radius[i] < r1:
                    values[i] = k1 * radius[i] ** 2
                elif radius[i] < r2:
                    values[i] = k2 * radius[i]
                else:
                    values[i] = k3 * radius[i] ** (1 / 2)

        self.terminal_velocity_body = terminal_velocity_body

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def a_w_ice_body(*, T_in, p_in, RH_in, qv_in, a_w_ice_out):
            for i in prange(T_in.shape[0]):  # pylint: disable=not-an-iterable
                pvi = pvi_C(T_in[i] - const.T0)
                pv = phys_pv(p_in[i], qv_in[i])
                pvs = pv / RH_in[i]
                a_w_ice_out[i] = pvi / pvs

        self.a_w_ice_body = a_w_ice_body

    def temperature_pressure_RH(self, *, rhod, thd, qv, T, p, RH):
        self.temperature_pressure_RH_body(
            rhod=rhod.data, thd=thd.data, qv=qv.data, T=T.data, p=p.data, RH=RH.data
        )

    def terminal_velocity(self, *, values, radius, k1, k2, k3, r1, r2):
        self.terminal_velocity_body(
            values=values, radius=radius, k1=k1, k2=k2, k3=k3, r1=r1, r2=r2
        )

    def explicit_euler(self, y, dt, dy_dt):
        self.explicit_euler_body(y.data, dt, dy_dt)

    def critical_volume(self, *, v_cr, kappa, f_org, v_dry, v_wet, T, cell):
        self.critical_volume_body(
            v_cr=v_cr.data,
            kappa=kappa.data,
            f_org=f_org.data,
            v_dry=v_dry.data,
            v_wet=v_wet.data,
            T=T.data,
            cell=cell.data,
        )

    def a_w_ice(self, *, T, p, RH, qv, a_w_ice):
        self.a_w_ice_body(
            T_in=T.data,
            p_in=p.data,
            RH_in=RH.data,
            qv_in=qv.data,
            a_w_ice_out=a_w_ice.data,
        )
