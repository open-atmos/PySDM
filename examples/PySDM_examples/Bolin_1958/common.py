import numpy as np
from PySDM.physics import si
from PySDM import Formulae

formulae = Formulae(
    terminal_velocity="RogersYau",
    drop_growth="Mason1951",
    diffusion_thermics="Neglect",
    saturation_vapour_pressure="AugustRocheMagnus",
    ventilation="Froessling1938",
    particle_shape_and_density="LiquidSpheres",
    air_dynamic_viscosity="ZografosEtAl1987",
    constants={"BOLIN_ISOTOPE_TIMESCALE_COEFF_C1": 1.63},
    isotope_relaxation_timescale="Bolin1958",
)
const = formulae.constants
any_non_zero_value = 44.0
radii = np.asarray([0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.20]) * si.cm
temperature = const.T0 + 10 * si.K
pressure = const.p_STP
pvs = formulae.saturation_vapour_pressure.pvs_water(temperature)
v_term = formulae.terminal_velocity.v_term(radii)
eta_air = formulae.air_dynamic_viscosity.eta_air(temperature)
D = formulae.diffusion_thermics.D(T=temperature, p=pressure)

air_density = pressure / const.Rd / temperature
assert abs(air_density - 1) / air_density < 0.3
Re = formulae.particle_shape_and_density.reynolds_number(
    radius=radii,
    velocity_wrt_air=v_term,
    dynamic_viscosity=eta_air,
    density=air_density,
)
Sc = formulae.trivia.air_schmidt_number(
    dynamic_viscosity=eta_air,
    diffusivity=D,
    density=air_density,
)
F = formulae.ventilation.ventilation_coefficient(
    sqrt_re_times_cbrt_sc=Re ** (1 / 2) * Sc ** (1 / 3)
)

r_dr_dt = formulae.drop_growth.r_dr_dt(
    RH_eq=1,
    T=temperature,
    RH=0,
    lv=0,
    pvs=pvs,
    D=D,
    K=any_non_zero_value,
    ventilation_factor=F,
)
