def r_dr_dt_fun(*, formulae, v_term, radii, temperature, K):

    pvs = formulae.saturation_vapour_pressure.pvs_water(temperature)
    eta_air = formulae.air_dynamic_viscosity.eta_air(temperature)
    const = formulae.constants
    pressure = const.p_STP
    D = formulae.diffusion_thermics.D(T=temperature, p=const.p_STP)

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
    sqrt_re_times_cbrt_sc = formulae.trivia.sqrt_re_times_cbrt_sc(Re=Re, Sc=Sc)
    F = formulae.ventilation.ventilation_coefficient(
        sqrt_re_times_cbrt_sc=sqrt_re_times_cbrt_sc
    )

    return formulae.drop_growth.r_dr_dt(
        RH_eq=1,
        T=temperature,
        RH=0,
        lv=0,
        pvs=pvs,
        D=D,
        K=K,
        ventilation_factor=F,
    )
