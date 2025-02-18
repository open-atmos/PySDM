from PySDM import Formulae


class IsotopeTimescale:
    def __init__(self, *, settings, temperature, radii):
        self.radii = radii
        self.formulae = Formulae(**settings)
        self.temperature = temperature
        self.pressure = self.formulae.constants.p_STP
        self.v_term = self.formulae.terminal_velocity.v_term(radii)
        self.D = self.formulae.diffusion_thermics.D(T=self.temperature, p=self.pressure)
        self.D_iso = self.formulae.isotope_diffusivity_ratios.ratio_3H(self.temperature)
        self.K = 44.0  # any non-zero value
        self.pvs_water = self.formulae.saturation_vapour_pressure.pvs_water(
            self.temperature
        )
        self.alpha = self.formulae.isotope_equilibrium_fractionation_factors.alpha_i_3H(
            self.temperature
        )  # check i/l
        self.M_iso = self.formulae.constants.M_3H

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

    def r_dr_dt(self, RH, RH_eq, lv):
        return self.formulae.drop_growth.r_dr_dt(
            T=self.temperature,
            pvs=self.pvs_water,
            D=self.D,
            K=self.K,
            ventilation_factor=self.vent_coeff_fun(),
            RH=RH,
            RH_eq=RH_eq,
            lv=lv,
        )

    def c1_coeff(self, *, vent_coeff_iso, rho_env, pvs_iso):
        return self.formulae.isotope_relaxation_timescale.c1_coeff(
            vent_coeff_iso=vent_coeff_iso,
            vent_coeff=self.vent_coeff_fun(),
            D_iso=self.D_iso,
            D=self.D,
            alpha=self.alpha,
            rho_env_iso=self.formulae.constants.VSMOW_R_3H,
            rho_env=rho_env,
            M_iso=self.M_iso,
            pvs_iso=pvs_iso,  # any number
            pvs_water=self.pvs_water,
            temperature=self.temperature,
        )
