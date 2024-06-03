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
    def _temperature_pressure_rh_body(self):
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
    def temperature_pressure_rh(
        self, *, rhod, thd, water_vapour_mixing_ratio, T, p, RH
    ):
        self._temperature_pressure_rh_body.launch_n(
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

    @cached_property
    def __air_density_body(self):
        return trtc.For(
            param_names=("output", "rhod", "water_vapour_mixing_ratio"),
            name_iter="i",
            body=f"""
            output[i] = {self.formulae.state_variable_triplet.rho_of_rhod_and_water_vapour_mixing_ratio.c_inline(
                rhod="rhod[i]",
                water_vapour_mixing_ratio="water_vapour_mixing_ratio[i]"
            )};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def air_density(self, output, rhod, water_vapour_mixing_ratio):
        self.__air_density_body.launch_n(
            n=output.shape[0],
            args=(output.data, rhod.data, water_vapour_mixing_ratio.data),
        )

    @cached_property
    def __air_dynamic_viscosity_body(self):
        return trtc.For(
            param_names=("output", "temperature"),
            name_iter="i",
            body=f"""
            output[i] = {self.formulae.air_dynamic_viscosity.eta_air.c_inline(
                temperature="temperature[i]"
            )};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def air_dynamic_viscosity(self, output, temperature):
        self.__air_dynamic_viscosity_body.launch_n(
            n=output.shape[0], args=(output.data, temperature.data)
        )

    @cached_property
    def __reynolds_number_body(self):
        return trtc.For(
            param_names=(
                "output",
                "cell_id",
                "air_dynamic_viscosity",
                "air_density",
                "radius",
                "velocity_wrt_air",
            ),
            name_iter="i",
            body=f"""
            output[i] = {self.formulae.particle_shape_and_density.reynolds_number.c_inline(
                radius="radius[i]",
                velocity_wrt_air="velocity_wrt_air[i]",
                dynamic_viscosity="air_dynamic_viscosity[cell_id[i]]",
                density="air_density[cell_id[i]]",
            )};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    def reynolds_number(
        self,
        *,
        output,
        cell_id,
        dynamic_viscosity,
        density,
        radius,
        velocity_wrt_air,
    ):
        self.__reynolds_number_body.launch_n(
            n=output.shape[0],
            args=(
                output.data,
                cell_id.data,
                dynamic_viscosity.data,
                density.data,
                radius.data,
                velocity_wrt_air.data,
            ),
        )
