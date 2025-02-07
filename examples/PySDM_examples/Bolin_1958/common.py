from functools import partial
from PySDM import Formulae


class IsotopeTimescaleCommon:
    def __init__(self, settings, temperature, radii):
        self.radii = radii
        self.formulae = Formulae(**settings)
        self.temperature = temperature
        self.pressure = self.formulae.constants.p_STP
        self.v_term = self.formulae.terminal_velocity.v_term(radii)
        self.D = self.formulae.diffusion_thermics.D(T=self.temperature, p=self.pressure)

    def vent_coeff_fun(self):
        eta_air = self.formulae.air_dynamic_viscosity.eta_air(self.temperature)
        air_density = self.pressure / self.formulae.constants.Rd / self.temperature

        assert abs(air_density - 1) / air_density < 0.3

        Re = self.formulae.particle_shape_and_density.reynolds_number(
            radius=self.radii,
            velocity_wrt_air=self.v_term,
            dynamic_viscosity=eta_air,
            density=air_density,
        )
        Sc = self.formulae.trivia.air_schmidt_number(
            dynamic_viscosity=eta_air,
            diffusivity=self.D,
            density=air_density,
        )

        return self.formulae.ventilation.ventilation_coefficient(
            sqrt_re_times_cbrt_sc=self.formulae.trivia.sqrt_re_times_cbrt_sc(
                Re=Re, Sc=Sc
            )
        )

    def r_dr_dt_fun(self, K):
        return partial(
            self.formulae.drop_growth.r_dr_dt,
            T=self.temperature,
            pvs=self.formulae.saturation_vapour_pressure.pvs_water(self.temperature),
            D=self.D,
            K=K,
            ventilation_factor=self.vent_coeff_fun(),
        )
