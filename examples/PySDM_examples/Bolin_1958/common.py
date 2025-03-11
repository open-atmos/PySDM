from PySDM import Formulae
from PySDM_examples.Szumowski_et_al_1998.sounding import temperature


class IsotopeTimescale:
    def __init__(self, *, settings, temperature, radii):
        self.radii = radii
        self.formulae = Formulae(**settings)
        self.temperature = temperature
        self.pressure = self.formulae.constants.p_STP
        self.v_term = self.formulae.terminal_velocity.v_term(radii)
        self.D = self.formulae.diffusion_thermics.D(T=self.temperature, p=self.pressure)
        self.K = self.formulae.diffusion_thermics.K(T=self.temperature, p=self.pressure)
        # self.K = 44.0  # any non-zero value
        self.pvs_water = self.formulae.saturation_vapour_pressure.pvs_water(
            self.temperature
        )
        self.alpha = self.formulae.isotope_equilibrium_fractionation_factors.alpha_l_3H(
            self.temperature
        )  # check i/l

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

    def k(self, lv, J, rho_s, R_v):
        return 1 / (
            1 / rho_s(self.temperature)
            + self.f * self.D * J * lv**2 / R_v / self.temperature**2 / self.K
        )

    # def M_rat(self, isotope):
    #    if (getattr(self.formulae.constants, f"M_{isotope}"))
    def c1(self, isotope, R_vap, RH, pv_iso, pv_water, alpha):
        return (
            -self.vent_coeff_fun()  # TODO f_iso!
            / getattr(self.formulae.isotope_diffusivity_ratios, f"ratio_{isotope}")(
                self.temperature
            )
            # * self.k_iso
            # / self.k
            / self.vent_coeff_fun()
            / (getattr(self.formulae.constants, f"M_{isotope}"))
            * (pv_iso / pv_water * RH / R_vap - 1)
            / alpha
            * (RH - 1)
            * R_vap
        )
