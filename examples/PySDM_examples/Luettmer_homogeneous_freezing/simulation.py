import numpy as np
import warnings
import PySDM.products as PySDM_products
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import (
    AmbientThermodynamics,
    Condensation,
    Freezing,
    VapourDepositionOnIce,
)
from PySDM.environments import Parcel
from PySDM.physics import constants as const
from PySDM.initialisation import discretise_multiplicities
from PySDM.initialisation.hygroscopic_equilibrium import equilibrate_wet_radii
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver


class Simulation:
    def __init__(self, settings, backend=CPU):

        dt = settings.dt

        formulae = settings.formulae

        env = Parcel(
            mixed_phase=True,
            dt=dt,
            mass_of_dry_air=settings.mass_of_dry_air,
            p0=settings.initial_pressure,
            initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
            T0=settings.initial_temperature,
            w=settings.w_updraft,
        )

        builder = Builder(
            backend=settings.backend,
            n_sd=settings.n_sd,
            environment=env,
        )

        builder.add_dynamic(AmbientThermodynamics())
        if settings.condensation_enable:
            builder.add_dynamic(Condensation(adaptive=True))
        if settings.deposition_enable:
            builder.add_dynamic(
                VapourDepositionOnIce(adaptive=settings.deposition_adaptive)
            )
        builder.add_dynamic(
            Freezing(
                homogeneous_freezing=settings.hom_freezing_type, immersion_freezing=None
            )
        )

        self.n_sd = settings.n_sd
        self.multiplicities = discretise_multiplicities(
            settings.specific_concentration * env.mass_of_dry_air
        )
        self.r_dry = settings.r_dry
        v_dry = settings.formulae.trivia.volume(radius=self.r_dry)
        kappa = settings.kappa

        self.r_wet = equilibrate_wet_radii(
            r_dry=self.r_dry,
            environment=builder.particulator.environment,
            kappa_times_dry_volume=kappa * v_dry,
        )
        self.initial_mass = formulae.particle_shape_and_density.radius_to_mass(
            self.r_wet
        )

        attributes = {
            "multiplicity": self.multiplicities,
            "dry volume": v_dry,
            "kappa times dry volume": kappa * v_dry,
            "signed water mass": self.initial_mass,
        }
        builder.request_attribute("temperature of last freezing")
        builder.request_attribute("radius")

        products = [
            PySDM_products.ParcelDisplacement(name="z"),
            PySDM_products.Time(name="t"),
            PySDM_products.AmbientRelativeHumidity(name="RH", unit="%"),
            PySDM_products.AmbientRelativeHumidity(name="RH_ice", unit="%"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.AmbientPressure(name="p", unit="hPa"),
            PySDM_products.WaterMixingRatio(name="water", radius_range=(0, np.inf)),
            PySDM_products.WaterMixingRatio(name="ice", radius_range=(-np.inf, 0)),
            PySDM_products.WaterMixingRatio(
                name="total", radius_range=(-np.inf, np.inf)
            ),
            PySDM_products.AmbientWaterVapourMixingRatio(
                name="vapour", var="water_vapour_mixing_ratio"
            ),
            PySDM_products.ParticleConcentration(name="n_s", radius_range=(0, np.inf)),
            PySDM_products.ParticleConcentration(name="n_i", radius_range=(-np.inf, 0)),
            PySDM_products.MeanRadius(name="r_s", radius_range=(0, np.inf)),
            PySDM_products.MeanRadius(name="r_i", radius_range=(-np.inf, 0)),
        ]

        self.particulator = builder.build(attributes, products)
        if settings.scipy_solver:
            scipy_ode_condensation_solver.patch_particulator(self.particulator)

        self.n_output = settings.n_output
        if settings.n_output == 1:
            self.n_substeps = 1
        else:
            self.n_substeps = int(self.n_output / dt)
        self.t_max_duration = settings.t_max_duration

    def save(self, output):
        cell_id = 0

        output["z"].append(self.particulator.products["z"].get()[cell_id])
        output["t"].append(self.particulator.products["t"].get())
        output["RH"].append(self.particulator.products["RH"].get()[cell_id])
        output["RHi"].append(self.particulator.products["RH_ice"].get()[cell_id])
        output["T"].append(self.particulator.products["T"].get()[cell_id])
        output["P"].append(self.particulator.products["p"].get()[cell_id])
        output["LWC"].append(self.particulator.products["water"].get()[cell_id])
        output["IWC"].append(self.particulator.products["ice"].get()[cell_id])
        output["qv"].append(self.particulator.products["vapour"].get()[cell_id])
        output["ns"].append(self.particulator.products["n_s"].get()[cell_id])
        output["ni"].append(self.particulator.products["n_i"].get()[cell_id])
        output["rs"].append(self.particulator.products["r_s"].get()[cell_id])
        output["ri"].append(self.particulator.products["r_i"].get()[cell_id])
        output["water_mass"].append(
            self.particulator.attributes["signed water mass"].data.tolist()
        )
        output["T_frz"].append(
            self.particulator.attributes["temperature of last freezing"].data.tolist()
        )

    def run(self):

        print("Starting simulation...")

        output = {
            "t": [],
            "z": [],
            "RH": [],
            "RHi": [],
            "T": [],
            "P": [],
            "LWC": [],
            "IWC": [],
            "qv": [],
            "ns": [],
            "ni": [],
            "rs": [],
            "ri": [],
            "water_mass": [],
            "T_frz": [],
        }

        self.save(output)

        while True:

            self.particulator.run(self.n_substeps)
            self.save(output)

            # print( output["t"][-1], output["T"][-1], output["RH"][-1], output["LWC"][-1], output["IWC"][-1] )

            if output["LWC"][-1] < output["LWC"][0]:
                print("all particles frozen or evaporated")
                # Assert for water saturation
                test_water_saturation = np.asarray(output["RH"])
                # Sort out times before CCN activation & after first occurence of ice
                test_water_saturation = np.where(
                    np.asarray(output["rs"]) < 1e-6, 100.0, test_water_saturation
                )
                test_water_saturation = np.where(
                    np.asarray(output["IWC"]) > 0.0, 100.0, test_water_saturation
                )
                if np.allclose(test_water_saturation, 100.0, rtol=5e-2) == False:
                    print(
                        "Warning: water saturation is too high outside of activation and mixed-phase environment"
                    )

                break
            if output["t"][-1] >= self.t_max_duration:
                print("time exceeded")
                break

        return output
