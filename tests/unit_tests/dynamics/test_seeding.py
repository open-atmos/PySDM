import numpy as np
from matplotlib import pyplot

from pystrict import strict

from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.environments import Parcel
from PySDM.dynamics import Condensation, AmbientThermodynamics, Coalescence, Seeding
from PySDM.dynamics.collisions.collision_kernels import Geometric
from PySDM.physics import si, in_unit
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.spectra import Lognormal


@strict
class Settings:
    def __init__(self):
        self.n_sd_initial = 100
        self.n_sd_seeding = 100
        self.t_max = 20 * si.min
        self.w_max = 3 * si.m / si.s
        self.w_min = 0.025 * si.m / si.s

        self.timestep = 15 * si.s
        self.mass_of_dry_air = 666 * si.kg

        self.updraft = (
            lambda t: self.w_min
            + (self.w_max - self.w_min)
            * np.maximum(0, np.sin(t / self.t_max * 2 * np.pi)) ** 2
        )
        self.initial_water_vapour_mixing_ratio = 666 / 30 * si.g / si.kg
        self.initial_total_pressure = 1000 * si.hPa
        self.initial_temperature = 300 * si.K
        self.initial_aerosol_kappa = 0.5
        self.initial_aerosol_dry_radii = Lognormal(
            norm_factor=200 / si.mg * self.mass_of_dry_air,
            m_mode=75 * si.nm,
            s_geom=1.6,
        )
        self.seeding_time_window = (10 * si.min, 15 * si.min)
        self.seeded_particle_multiplicity = 1
        self.seeded_particle_extensive_attributes = {
            "water mass": 0.001 * si.ng,
            "dry volume": 0.0001 * si.ng,
            "kappa times dry volume": 0.8 * 0.0001 * si.ng,
        }


def test_seeding(plot=True):
    # arrange
    settings = Settings()
    builder = Builder(
        n_sd=settings.n_sd_seeding + settings.n_sd_initial,
        backend=CPU(formulae=Formulae(seed=100)),
        environment=Parcel(
            dt=settings.timestep,
            mass_of_dry_air=settings.mass_of_dry_air,
            w=settings.updraft,
            initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
            p0=settings.initial_total_pressure,
            T0=settings.initial_temperature,
        ),
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())
    builder.add_dynamic(Coalescence(collision_kernel=Geometric()))
    builder.add_dynamic(
        Seeding(
            time_window=settings.seeding_time_window,
            seeded_particle_multiplicity=settings.seeded_particle_multiplicity,
            seeded_particle_extensive_attributes=settings.seeded_particle_extensive_attributes,
        )
    )

    r_dry, n_in_dv = ConstantMultiplicity(settings.initial_aerosol_dry_radii).sample(
        n_sd=settings.n_sd_initial, backend=builder.particulator.backend
    )
    attributes = builder.particulator.environment.init_attributes(
        n_in_dv=n_in_dv, kappa=settings.initial_aerosol_kappa, r_dry=r_dry
    )
    particulator = builder.build(
        attributes={
            k: np.pad(
                array=v,
                pad_width=(0, settings.n_sd_seeding),
                mode="constant",
                constant_values=np.nan if k == "multiplicity" else 0,
            )
            for k, v in attributes.items()
        }
    )

    # act
    output = []
    for step in range(int(settings.t_max // settings.timestep) + 1):
        if step != 0:
            particulator.run(steps=1)
        output.append(particulator.attributes["water mass"].to_ndarray(raw=True))
    output = np.array(output)

    # plot
    time = np.linspace(start=0, stop=settings.t_max, num=output.shape[0])
    for drop_id in range(output.shape[1]):
        pyplot.plot(
            in_unit(output[:, drop_id], si.ng),
            in_unit(time, si.min),
            color="navy" if output[0, drop_id] != 0 else "red",
            linewidth=0.333,
        )
    pyplot.ylabel("time [min]")
    pyplot.xlabel("drop mass [ng]")
    pyplot.grid()
    pyplot.xscale("log")
    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    # TODO
