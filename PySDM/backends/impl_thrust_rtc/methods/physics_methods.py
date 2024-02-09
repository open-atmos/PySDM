"""
GPU implementation of backend methods wrapping basic physics formulae
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class PhysicsMethods(ThrustRTCBackendMethods):
    @cached_property
    def _temperature_pressure_RH_body(self):
        return trtc.For(
            ("rhod", "thd", "water_vapour_mixing_ratio", "T", "p", "RH"),
            "i",
            f"""
            T[i] = {self.formulae.state_variable_triplet.T.c_inline(
                rhod="rhod[i]", thd="thd[i]")};
            p[i] = {self.formulae.state_variable_triplet.p.c_inline(
                rhod="rhod[i]",
                T="T[i]",
                water_vapour_mixing_ratio="water_vapour_mixing_ratio[i]"
            )};
            RH[i] = {self.formulae.state_variable_triplet.pv.c_inline(
                p="p[i]", water_vapour_mixing_ratio="water_vapour_mixing_ratio[i]"
            )} / {self.formulae.saturation_vapour_pressure.pvs_Celsius.c_inline(
                T="T[i] - const.T0"
            )};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __explicit_euler_body(self):
        return trtc.For(
            ("y", "dt", "dy_dt"),
            "i",
            f"""
            y[i] = {self.formulae.trivia.explicit_euler.c_inline(y="y[i]", dt="dt", dy_dt="dy_dt")};
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __critical_volume_body(self):
        return trtc.For(
            ("v_cr", "kappa", "f_org", "v_dry", "v_wet", "T", "cell"),
            "i",
            f"""
            auto sigma = {self.formulae.surface_tension.sigma.c_inline(
                T="T[cell[i]]", v_wet="v_wet[i]", v_dry="v_dry[i]", f_org="f_org[i]"
            )};
            auto r_cr = {self.formulae.hygroscopicity.r_cr.c_inline(
                kp="kappa[i]",
                rd3="v_dry[i] / const.PI_4_3",
                T="T[cell[i]]",
                sgm="sigma"
            )};
            v_cr[i] = {self.formulae.trivia.volume.c_inline(radius="r_cr")};
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __terminal_velocity_body(self):
        return trtc.For(
            ("values", "radius", "k1", "k2", "k3", "r1", "r2"),
            "i",
            """
            if (radius[i] < r1) {
                values[i] = k1 * radius[i] * radius[i];
            }
            else {
                if (radius[i] < r2) {
                    values[i] = k2 * radius[i];
                }
                else {
                    values[i] = k3 * pow(radius[i], (real_type)(.5));
                }
            }
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __volume_of_mass_body(self):
        return trtc.For(
            param_names=("volume", "mass"),
            name_iter="i",
            body=f"""
            volume[i] = {self.formulae.particle_shape_and_density.mass_to_volume.c_inline(mass="mass[i]")};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __mass_of_volume_body(self):
        return trtc.For(
            param_names=("mass", "volume"),
            name_iter="i",
            body=f"""
            mass[i] = {self.formulae.particle_shape_and_density.volume_to_mass.c_inline(volume="volume[i]")};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def critical_volume(self, *, v_cr, kappa, f_org, v_dry, v_wet, T, cell):
        self.__critical_volume_body.launch_n(
            v_cr.shape[0],
            (
                v_cr.data,
                kappa.data,
                f_org.data,
                v_dry.data,
                v_wet.data,
                T.data,
                cell.data,
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def temperature_pressure_RH(
        self, *, rhod, thd, water_vapour_mixing_ratio, T, p, RH
    ):
        self._temperature_pressure_RH_body.launch_n(
            T.shape[0],
            (
                rhod.data,
                thd.data,
                water_vapour_mixing_ratio.data,
                T.data,
                p.data,
                RH.data,
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def terminal_velocity(self, *, values, radius, k1, k2, k3, r1, r2):
        k1 = self._get_floating_point(k1)
        k2 = self._get_floating_point(k2)
        k3 = self._get_floating_point(k3)
        r1 = self._get_floating_point(r1)
        r2 = self._get_floating_point(r2)
        self.__terminal_velocity_body.launch_n(
            values.size(), [values, radius, k1, k2, k3, r1, r2]
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def explicit_euler(self, y, dt, dy_dt):
        dt = self._get_floating_point(dt)
        dy_dt = self._get_floating_point(dy_dt)
        self.__explicit_euler_body.launch_n(y.shape[0], (y.data, dt, dy_dt))

    @nice_thrust(**NICE_THRUST_FLAGS)
    def volume_of_water_mass(self, volume, mass):
        self.__volume_of_mass_body.launch_n(volume.shape[0], (volume.data, mass.data))

    @nice_thrust(**NICE_THRUST_FLAGS)
    def mass_of_water_volume(self, mass, volume):
        self.__mass_of_volume_body.launch_n(mass.shape[0], (mass.data, volume.data))
