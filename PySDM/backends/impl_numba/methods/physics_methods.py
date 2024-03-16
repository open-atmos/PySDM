"""
CPU implementation of backend methods wrapping basic physics formulae
"""

import numba
from numba import prange

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf


class PhysicsMethods(BackendMethods):
    def __init__(self):  # pylint: disable=too-many-locals
        BackendMethods.__init__(self)
        pvs_C = self.formulae.saturation_vapour_pressure.pvs_Celsius
        pvi_C = self.formulae.saturation_vapour_pressure.ice_Celsius
        phys_T = self.formulae.state_variable_triplet.T
        phys_p = self.formulae.state_variable_triplet.p
        phys_pv = self.formulae.state_variable_triplet.pv
        explicit_euler = self.formulae.trivia.explicit_euler
        phys_sigma = self.formulae.surface_tension.sigma
        phys_volume = self.formulae.trivia.volume
        phys_r_cr = self.formulae.hygroscopicity.r_cr
        phys_mass_to_volume = self.formulae.particle_shape_and_density.mass_to_volume
        phys_volume_to_mass = self.formulae.particle_shape_and_density.volume_to_mass
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
        def temperature_pressure_RH_body(
            *, rhod, thd, water_vapour_mixing_ratio, T, p, RH
        ):
            for i in prange(T.shape[0]):  # pylint: disable=not-an-iterable
                T[i] = phys_T(rhod[i], thd[i])
                p[i] = phys_p(rhod[i], T[i], water_vapour_mixing_ratio[i])
                RH[i] = phys_pv(p[i], water_vapour_mixing_ratio[i]) / pvs_C(
                    T[i] - const.T0
                )

        self.temperature_pressure_RH_body = temperature_pressure_RH_body

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def a_w_ice_body(
            *, T_in, p_in, RH_in, water_vapour_mixing_ratio_in, a_w_ice_out
        ):
            for i in prange(T_in.shape[0]):  # pylint: disable=not-an-iterable
                pvi = pvi_C(T_in[i] - const.T0)
                pv = phys_pv(p_in[i], water_vapour_mixing_ratio_in[i])
                pvs = pv / RH_in[i]
                a_w_ice_out[i] = pvi / pvs

        self.a_w_ice_body = a_w_ice_body

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def volume_of_mass(volume, mass):
            for i in prange(volume.shape[0]):  # pylint: disable=not-an-iterable
                volume[i] = phys_mass_to_volume(mass[i])

        self.volume_of_mass_body = volume_of_mass

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def mass_of_volume(mass, volume):
            for i in prange(volume.shape[0]):  # pylint: disable=not-an-iterable
                mass[i] = phys_volume_to_mass(volume[i])

        self.mass_of_volume_body = mass_of_volume

    def temperature_pressure_RH(
        self, *, rhod, thd, water_vapour_mixing_ratio, T, p, RH
    ):
        self.temperature_pressure_RH_body(
            rhod=rhod.data,
            thd=thd.data,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio.data,
            T=T.data,
            p=p.data,
            RH=RH.data,
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

    def a_w_ice(self, *, T, p, RH, water_vapour_mixing_ratio, a_w_ice):
        self.a_w_ice_body(
            T_in=T.data,
            p_in=p.data,
            RH_in=RH.data,
            water_vapour_mixing_ratio_in=water_vapour_mixing_ratio.data,
            a_w_ice_out=a_w_ice.data,
        )

    def volume_of_water_mass(self, volume, mass):
        self.volume_of_mass_body(volume.data, mass.data)

    def mass_of_water_volume(self, mass, volume):
        self.mass_of_volume_body(mass.data, volume.data)
