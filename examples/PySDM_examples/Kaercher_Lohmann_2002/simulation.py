import numpy as np

import PySDM.products as PySDM_products
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import AmbientThermodynamics, Condensation, Freezing, VapourDepositionOnIce
from PySDM.environments import Parcel
from PySDM.physics import constants as const
from PySDM.initialisation import discretise_multiplicities, equilibrate_wet_radii


class Simulation:
    def __init__(self, settings, backend=CPU):
        # t_half = settings.z_half / settings.w_avg

        # dt_output = (2 * t_half) / settings.n_output
        # self.n_substeps = 1
        # while dt_output / self.n_substeps >= settings.dt_max:  # TODO #334 dt_max
        #     self.n_substeps += 1

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
            backend=backend(
                formulae=settings.formulae,
                **(
                    {"override_jit_flags": {"parallel": False}}
                    if backend == CPU
                    else {}
                )
            ),
            n_sd=settings.n_sd,
            environment=env,
        )

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())
        builder.add_dynamic(VapourDepositionOnIce())
        builder.add_dynamic(Freezing(singular=False, homogeneous_freezing=True, immersion_freezing=False))

        self.n_sd = settings.n_sd
        self.multiplicities = discretise_multiplicities(settings.specific_concentration * env.mass_of_dry_air)
        self.r_dry = settings.r_dry
        v_dry = settings.formulae.trivia.volume(radius=self.r_dry)
        kappa = settings.kappa

        self.r_wet = equilibrate_wet_radii(r_dry=self.r_dry, environment=builder.particulator.environment,
                                           kappa_times_dry_volume=kappa * v_dry)

        attributes = {
            "multiplicity": self.multiplicities,
            'dry volume': v_dry,
            'kappa times dry volume': kappa * v_dry,
            # 'volume': formulae.trivia.volume(radius=self.r_wet),
            "signed water mass": formulae.particle_shape_and_density.radius_to_mass(self.r_wet),
        }

        products = [
            PySDM_products.ParcelDisplacement(name="z"),
            PySDM_products.Time(name="t"),
            PySDM_products.AmbientRelativeHumidity(name="RH", unit="%"),
            PySDM_products.AmbientRelativeHumidity(name="RH_ice", unit="%"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.AmbientPressure(name="p", unit='hPa'),
            PySDM_products.WaterMixingRatio(name="water", radius_range=(0, np.inf)),
            PySDM_products.WaterMixingRatio(name="ice", radius_range=(-np.inf, 0)),
            PySDM_products.WaterMixingRatio(name="total", radius_range=(-np.inf, np.inf)),
            PySDM_products.AmbientWaterVapourMixingRatio(
                name="vapour", var="water_vapour_mixing_ratio"
            ),
            PySDM_products.ParticleConcentration(
                name='n_s', unit='1/cm**3',
                radius_range=(0, np.inf)),
            PySDM_products.ParticleConcentration(
                name='n_i', unit='1/cm**3',
                radius_range=(-np.inf, 0)),
            PySDM_products.MeanRadius(
                name='r_s', unit='µm',
                radius_range=(0, np.inf)),
            PySDM_products.MeanRadius(
                name='r_i', unit='µm',
                radius_range=(-np.inf, 0)),
        ]

        self.particulator = builder.build(attributes, products)

        self.n_output = settings.n_output
        self.n_substeps = int(settings.t_duration / dt / self.n_output)


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
        #   output["TWC"].append(self.particulator.products["total"].get()[cell_id])
        output["qv"].append(self.particulator.products["vapour"].get()[cell_id])
        output["ns"].append(self.particulator.products["n_s"].get()[cell_id])
        output["ni"].append(self.particulator.products["n_i"].get()[cell_id])
        output["rs"].append(self.particulator.products["r_s"].get()[cell_id])
        output["ri"].append(self.particulator.products["r_i"].get()[cell_id])
        output["water_mass"].append(self.particulator.attributes["signed water mass"].data.tolist())
    def run(self):
        output = {
            "t": [],
            "z": [],
            "RH": [],
            "RHi": [],
            "T": [],
            "P": [],
            "LWC": [],
            "IWC": [],
            #        "TWC": [],
            "qv": [],
            "ns": [],
            "ni": [],
            "rs": [],
            "ri": [],
            "frozen":[],
            "multiplicity": [],
            "r_dry":[],
            "r_wet": [],
            "water_mass":[]
        }

        self.save(output)
        output["n_sd"] =  [ self.n_sd ]
        output["r_dry"].append( self.r_dry.tolist() )
        output["r_wet"].append(self.r_wet.tolist())

        RHi_old = self.particulator.products["RH_ice"].get()[0].copy()
        for _ in range(self.n_output):
            # print(self.particulator.__dict__)
            # print(self.particulator.attributes.__dict__)
            # print(self.particulator.attributes._ParticleAttributes__attributes['signed water mass'])
            self.particulator.run(self.n_substeps)
            # print(self.particulator.products["t"].get())
            # print(self.particulator.products["p"].get())
            #
            self.save(output)


            RHi = self.particulator.products["RH_ice"].get()[0].copy()
            dRHi = (RHi_old - RHi) / RHi_old
            if (dRHi > 0. and RHi < 130.):
                print("break")
                break
            else:
                RHi_old = RHi

        frozen = np.where(self.particulator.attributes["signed water mass"].data < 0., 1, 0)
        output["frozen"].append(frozen.tolist())
        output["multiplicity"].append(
            self.particulator.attributes["multiplicity"].data.tolist()
        )
        print("Done. Last ni:", output["ni"][-1] )

        return output
