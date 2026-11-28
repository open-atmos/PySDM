import numpy as np
import matplotlib.pyplot as plt
from PySDM import products, Particulator, Formulae
from PySDM_examples.featured.Arabas_et_al_2025.run_simulation import update_thermo
from PySDM.backends import Numba
from PySDM_examples.sandbox.Abade_and_Albuquerque_2024.sampling import sampling_new
from PySDM.physics import si
from PySDM_examples.featured.Arabas_et_al_2025.commons import FREEZING_CONSTANTS
from PySDM.environments import Box
from PySDM.dynamics import Freezing


def run_simulation(particulator, T0, T_end, cooling_rate):
    dt = particulator.dt
    n_steps = int(np.ceil((T0 - T_end) / (cooling_rate * dt)))

    temperatures = [particulator.products["T"].get()[0]]
    frozen_fractions = [0.0]

    event_temperatures = np.full(particulator.n_sd, np.nan)

    signed_mass = particulator.attributes["signed water mass"].to_ndarray()
    frozen_before = signed_mass < 0

    for step in range(n_steps):
        T_now = T0 - (step + 1) * dt * cooling_rate
        update_thermo(particulator.environment, T=T_now)
        particulator.advance(1)

        signed_mass = particulator.attributes["signed water mass"].to_ndarray()
        frozen_now = signed_mass < 0

        just_frozen = frozen_now & ~frozen_before
        event_temperatures[just_frozen] = T_now
        frozen_before = frozen_now.copy()

        T = particulator.products["T"].get()[0]
        qi = particulator.products["ice water content"].get()[0]
        q_total = particulator.products["total water content"].get()[0]

        temperatures.append(T)
        frozen_fractions.append(qi / q_total if q_total > 0 else 0.0)

    return np.asarray(temperatures), np.asarray(frozen_fractions), event_temperatures


def make_particulator_case(
    *,
    singular,
    seed,
    T0,
    n_sd=1000,
    freezing_inp_frac=0.5,
    freezing_inp_dry_radius=10.0 * si.um,
    inp_surface_sigma=2.5,
    dt=1.0 * si.s,
    dv=1.0 * si.m**3,
    number_concentration=100 / si.l,
    collision=None,
):
    formulae_case = Formulae(
        seed=seed,
        constants={
            **FREEZING_CONSTANTS["dust"],
        },
        freezing_temperature_spectrum="Niemand_et_al_2012",
        heterogeneous_ice_nucleation_rate="ABIFM",
        particle_shape_and_density="MixedPhaseSpheres",
    )
    backend_case = Numba(formulae=formulae_case)

    attributes = {}
    A_sd = sampling_new(
        n_sd=n_sd,
        freezing_inp_frac=freezing_inp_frac,
        rng=np.random.default_rng(seed),
        singular=singular,
        freezing_inp_dry_radius=freezing_inp_dry_radius,
        attributes=attributes,
        formulae=formulae_case,
        inp_surface_sigma=inp_surface_sigma,
    )

    rho_air = 1000 * si.hPa / formulae_case.constants.Rd / (300 * si.K)
    if singular:
        attributes["immersed surface area"] = np.nan_to_num(
            A_sd.copy(),
            nan=0.0,
        )
    attributes["multiplicity"] = np.full(
        n_sd,
        number_concentration * dv / n_sd * rho_air / formulae_case.constants.rho_STP,
    )
    attributes["signed water mass"] = np.full(
        n_sd,
        1.0 * si.ug,
    )

    environment = Box(dt=dt, dv=dv, backend=backend_case)
    environment["T"] = T0
    update_thermo(environment, T=T0)
    environment["RH"] = 1.2
    environment["rhod"] = rho_air
    dynamics = (
        Freezing(immersion_freezing="singular" if singular else "time-dependent"),
    )

    if collision is not None:
        dynamics += (collision,)

    output_products = (
        products.Time(name="time"),
        products.AmbientTemperature(name="T"),
        products.IceWaterContent(name="ice water content"),
        products.CloudWaterContent(name="total water content"),
    )

    return Particulator(
        n_sd=n_sd,
        attributes=attributes,
        environment=environment,
        dynamics=dynamics,
        products=output_products,
        requested_attributes=(
            "radius",
            "multiplicity",
        ),
    )


def rmse_between_curves(T_sing, f_sing, T_td, f_td, Tmin=238, Tmax=264):
    T_grid = np.linspace(Tmin, Tmax, 300)

    f_sing_grid = np.interp(
        T_grid,
        (T_sing / si.K)[::-1],
        f_sing[::-1],
    )
    f_td_grid = np.interp(
        T_grid,
        (T_td / si.K)[::-1],
        f_td[::-1],
    )

    return np.sqrt(np.mean((f_sing_grid - f_td_grid) ** 2))


def draw_background(fig, ax, area, temperature, joint_pdf):
    plot = ax.pcolormesh(
        area / si.um**2,
        temperature / si.K,
        joint_pdf * si.um**2 * si.K,
        shading="auto",
        cmap="viridis",
    )
    ax.set(
        xscale="log",
        xlabel="Immersed INP surface area [µm²]",
        ylabel="Freezing temperature [K]",
        title="Joint surface-area and freezing-temperature distribution",
    )
    fig.colorbar(plot, ax=ax, label="Joint probability density [µm⁻² K⁻¹]")


def make_freezing_frame(
    particulator,
    area,
    temperature,
    joint_pdf,
    T0,
    cooling_rate,
    singular,
):
    T_homogeneous = 235 * si.K
    dt = particulator.dt

    signed_mass = particulator.attributes["signed water mass"].to_ndarray()

    event_temperatures = np.full_like(
        signed_mass,
        np.nan,
        dtype=float,
    )

    frozen_before = signed_mass < 0
    current_step = 0
    T_now = T0

    A_sd = particulator.attributes["immersed surface area"].to_ndarray()

    T_freeze_sd = None

    if singular:
        T_freeze_sd = particulator.attributes["freezing temperature"].to_ndarray()

        valid = (A_sd > 0) & (T_freeze_sd > T_homogeneous)

    else:
        valid = A_sd > 0

    frozen = (signed_mass < 0) & valid
    unfrozen = valid & ~frozen

    def plot_func(i, ax=None):
        nonlocal current_step, T_now, T_homogeneous
        nonlocal signed_mass, A_sd, valid, frozen, unfrozen, T_freeze_sd
        while current_step < i:

            T_now = T0 - (current_step + 1) * dt * cooling_rate
            update_thermo(particulator.environment, T=T_now)
            particulator.advance(1)
            signed_mass = particulator.attributes["signed water mass"].to_ndarray()

            A_sd = particulator.attributes["immersed surface area"].to_ndarray()

            if singular:
                T_freeze_sd = particulator.attributes[
                    "freezing temperature"
                ].to_ndarray()

                valid = (A_sd > 0) & (T_freeze_sd > T_homogeneous)

            else:
                valid = A_sd > 0
            frozen = (signed_mass < 0) & valid
            unfrozen = valid & ~frozen
            if not singular:
                just_frozen = frozen & ~frozen_before
                event_temperatures[just_frozen] = T_now

            frozen_before[:] = frozen
            current_step += 1

        if singular:
            y_unfrozen = T_freeze_sd[unfrozen] / si.K
            y_frozen = T_freeze_sd[frozen] / si.K
            title = "Singular freezing of super-droplets"
        else:
            y_unfrozen = np.full(unfrozen.sum(), T_now / si.K)
            y_frozen = event_temperatures[frozen] / si.K
            title = "Time-dependent freezing of super-droplets"
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        else:
            fig = ax.get_figure()
        draw_background(fig, ax, area, temperature, joint_pdf)

        ax.scatter(
            A_sd[unfrozen] / si.um**2,
            y_unfrozen,
            color="mediumseagreen",
            s=14,
            alpha=0.7,
            label="Liquid super-droplets",
        )
        ax.scatter(
            A_sd[frozen] / si.um**2,
            y_frozen,
            color="crimson",
            s=14,
            alpha=0.85,
            label="Frozen super-droplets",
        )
        ax.axhline(T_now / si.K, color="white", linestyle="--", linewidth=1.2)

        ax.set_title(title)
        ax.legend(loc="upper left")
        return fig

    return plot_func
