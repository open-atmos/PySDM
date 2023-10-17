import numpy as np
from PySDM_examples.Morrison_and_Grabowski_2007.common import Common
from PySDM_examples.Szumowski_et_al_1998 import sounding
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from PySDM.physics import si


class Cumulus(Common):
    def __init__(self, formulae):
        super().__init__(formulae)
        self.size = (9 * si.km, 2.7 * si.km)
        self.hx = 1.8 * si.km
        self.x0 = 3.6 * si.km
        self.xc = self.size[0] / 2
        self.dz = 50 * si.m
        self.grid = tuple(s / self.dz for s in self.size)
        for g in self.grid:
            assert int(g) == g
        self.grid = tuple(int(g) for g in self.grid)
        self.dt = 1 * si.s
        self.simulation_time = 60 * si.min

        self.__init_profiles()

    def __init_profiles(self):
        z_of_p = self.__z_of_p()
        T_of_z = interp1d(z_of_p, sounding.temperature)
        p_of_z = interp1d(z_of_p, sounding.pressure)
        q_of_z = interp1d(z_of_p, sounding.mixing_ratio)

        z_points_scalar = np.linspace(
            self.dz / 2, self.size[-1] - self.dz / 2, num=self.grid[-1], endpoint=True
        )
        z_points_both = np.linspace(
            0, self.size[-1], num=2 * self.grid[-1] + 1, endpoint=True
        )
        rhod_of_z = self.__rhod_of_z(T_of_z, p_of_z, q_of_z, z_of_p, z_points_both)
        self.rhod = interp1d(z_points_both / self.size[-1], rhod_of_z)
        self.initial_water_vapour_mixing_ratio = q_of_z(z_points_scalar)
        self.th_std0 = self.formulae.trivia.th_std(
            p_of_z(z_points_scalar), T_of_z(z_points_scalar)
        )

    def rhod_of_zZ(self, zZ):
        return self.rhod(zZ)

    def __rhod_of_z(self, T_of_z, p_of_z, q_of_z, z_of_p, z_points):
        def drhod_dz(z, _):
            lv = self.formulae.latent_heat.lv(T_of_z(z))
            return self.formulae.hydrostatics.drho_dz(
                self.formulae.constants.g_std, p_of_z(z), T_of_z(z), q_of_z(z), lv
            )

        theta_std0 = self.formulae.trivia.th_std(
            sounding.pressure[0], sounding.temperature[0]
        )
        rhod0 = self.formulae.state_variable_triplet.rho_d(
            sounding.pressure[0], sounding.mixing_ratio[0], theta_std0
        )
        rhod_solution = solve_ivp(
            fun=drhod_dz,
            t_span=(0, np.amax(z_of_p)),
            y0=np.asarray((rhod0,)),
            t_eval=z_points,
        )
        assert rhod_solution.success
        return rhod_solution.y[0]

    def __z_of_p(self):
        T_virt_of_p = interp1d(
            sounding.pressure,
            sounding.temperature
            * (1 + sounding.mixing_ratio / self.formulae.constants.eps)
            / (1 + sounding.mixing_ratio),
        )

        def dz_dp(p, _):
            return (
                -self.formulae.constants.Rd
                * T_virt_of_p(p)
                / self.formulae.constants.g_std
                / p
            )

        z_of_p = solve_ivp(
            fun=dz_dp,
            t_span=(max(sounding.pressure), min(sounding.pressure)),
            y0=np.asarray((0,)),
            t_eval=sounding.pressure,
        )
        assert z_of_p.success
        return z_of_p.y[0]

    @staticmethod
    def z0(z):
        return np.where(z <= 1.7 * si.km, 0, 0.7 * si.km)

    @staticmethod
    def hz(z):
        return np.where(z <= 1.7 * si.km, 3.4 * si.km, 2.0 * si.km)

    def alpha(self, x):
        return np.where(abs(x - self.xc) <= 0.9 * si.km, 1, 0)

    @staticmethod
    def beta(x):
        return np.where(x <= 5.4 * si.km, 1, -1)

    @staticmethod
    def A1(t):
        if t <= 900 * si.s:
            return 5.73e2
        if t <= 1500 * si.s:
            return 5.73e2 + 2.02e3 * (1 + np.cos(np.pi * ((t - 900) / 600 + 1)))
        return 1.15e3 + 1.72e3 * (1 + np.cos(np.pi * ((min(t, 2400) - 1500) / 900 + 1)))

    @staticmethod
    def A2(t):
        if t <= 300 * si.s:
            return 0
        if t <= 1500 * si.s:
            return 6e2 * (1 + np.cos(np.pi * ((t - 300) / 600 - 1)))
        return 5e2 * (1 + np.cos(np.pi * ((min(2400, t) - 1500) / 900 - 1)))

    # see Appendix (page 2859)
    def stream_function(self, xX, zZ, t):
        x = xX * self.size[0]
        z = zZ * self.size[-1]
        return (
            -self.A1(t)
            * np.cos(self.alpha(x) * np.pi * (x - self.x0) / self.hx)
            * np.sin(self.beta(x) * np.pi * (z - self.z0(z)) / self.hz(z))
            + self.A2(t) / 2 * zZ**2
        )
