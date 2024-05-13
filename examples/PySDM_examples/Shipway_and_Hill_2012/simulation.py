from collections import namedtuple

import numpy as np
from PySDM_examples.Shipway_and_Hill_2012.mpdata_1d import MPDATA_1D

import PySDM.products as PySDM_products
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import (
    AmbientThermodynamics,
    Coalescence,
    Condensation,
    Displacement,
    EulerianAdvection,
)
from PySDM.environments.kinematic_1d import Kinematic1D
from PySDM.impl.mesh import Mesh
from PySDM.initialisation.sampling import spatial_sampling, spectral_sampling


class Simulation:
    def __init__(self, settings, backend=CPU):
        self.nt = settings.nt
        self.z0 = -settings.particle_reservoir_depth
        self.save_spec_and_attr_times = settings.save_spec_and_attr_times
        self.number_of_bins = settings.number_of_bins

        self.particulator = None
        self.output_attributes = None
        self.output_products = None

        self.mesh = Mesh(
            grid=(settings.nz,),
            size=(settings.z_max + settings.particle_reservoir_depth,),
        )

        self.env = Kinematic1D(
            dt=settings.dt,
            mesh=self.mesh,
            thd_of_z=settings.thd,
            rhod_of_z=settings.rhod,
            z0=-settings.particle_reservoir_depth,
        )

        def zZ_to_z_above_reservoir(zZ):
            z_above_reservoir = zZ * (settings.nz * settings.dz) + self.z0
            return z_above_reservoir

        self.mpdata = MPDATA_1D(
            nz=settings.nz,
            dt=settings.dt,
            mpdata_settings=settings.mpdata_settings,
            advector_of_t=lambda t: settings.rho_times_w(t) * settings.dt / settings.dz,
            advectee_of_zZ_at_t0=lambda zZ: settings.water_vapour_mixing_ratio(
                zZ_to_z_above_reservoir(zZ)
            ),
            g_factor_of_zZ=lambda zZ: settings.rhod(zZ_to_z_above_reservoir(zZ)),
        )

        _extra_nz = settings.particle_reservoir_depth // settings.dz
        _z_vec = settings.dz * np.linspace(
            -_extra_nz, settings.nz - _extra_nz, settings.nz + 1
        )
        self.g_factor_vec = settings.rhod(_z_vec)

        self.builder = Builder(
            n_sd=settings.n_sd,
            backend=backend(formulae=settings.formulae),
            environment=self.env,
        )
        self.builder.add_dynamic(AmbientThermodynamics())

        if settings.enable_condensation:
            self.builder.add_dynamic(
                Condensation(
                    adaptive=settings.condensation_adaptive,
                    rtol_thd=settings.condensation_rtol_thd,
                    rtol_x=settings.condensation_rtol_x,
                    update_thd=settings.condensation_update_thd,
                )
            )
        self.builder.add_dynamic(EulerianAdvection(self.mpdata))

        self.products = []
        if settings.precip:
            self.add_collision_dynamic(self.builder, settings, self.products)

        displacement = Displacement(
            enable_sedimentation=settings.precip,
            precipitation_counting_level_index=int(
                settings.particle_reservoir_depth / settings.dz
            ),
        )
        self.builder.add_dynamic(displacement)
        self.attributes = self.env.init_attributes(
            spatial_discretisation=spatial_sampling.Pseudorandom(),
            spectral_discretisation=spectral_sampling.ConstantMultiplicity(
                spectrum=settings.wet_radius_spectrum_per_mass_of_dry_air
            ),
            kappa=settings.kappa,
            collisions_only=not settings.enable_condensation,
            z_part=settings.z_part,
        )
        self.products += [
            PySDM_products.WaterMixingRatio(
                name="cloud water mixing ratio",
                unit="g/kg",
                radius_range=settings.cloud_water_radius_range,
            ),
            PySDM_products.WaterMixingRatio(
                name="rain water mixing ratio",
                unit="g/kg",
                radius_range=settings.rain_water_radius_range,
            ),
            PySDM_products.AmbientDryAirDensity(name="rhod"),
            PySDM_products.AmbientDryAirPotentialTemperature(name="thd"),
            PySDM_products.ParticleSizeSpectrumPerVolume(
                name="wet spectrum", radius_bins_edges=settings.r_bins_edges
            ),
            PySDM_products.ParticleConcentration(
                name="nc", radius_range=settings.cloud_water_radius_range
            ),
            PySDM_products.ParticleConcentration(
                name="nr", radius_range=settings.rain_water_radius_range
            ),
            PySDM_products.ParticleConcentration(
                name="na", radius_range=(0, settings.cloud_water_radius_range[0])
            ),
            PySDM_products.MeanRadius(),
            PySDM_products.EffectiveRadius(
                radius_range=settings.cloud_water_radius_range
            ),
            PySDM_products.SuperDropletCountPerGridbox(),
            PySDM_products.AveragedTerminalVelocity(
                name="rain averaged terminal velocity",
                radius_range=settings.rain_water_radius_range,
            ),
            PySDM_products.AmbientRelativeHumidity(name="RH", unit="%"),
            PySDM_products.AmbientPressure(name="p"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.AmbientWaterVapourMixingRatio(
                name="water_vapour_mixing_ratio"
            ),
        ]
        if settings.enable_condensation:
            self.products.extend(
                [
                    PySDM_products.RipeningRate(name="ripening"),
                    PySDM_products.ActivatingRate(name="activating"),
                    PySDM_products.DeactivatingRate(name="deactivating"),
                    PySDM_products.PeakSupersaturation(unit="%"),
                    PySDM_products.ParticleSizeSpectrumPerVolume(
                        name="dry spectrum",
                        radius_bins_edges=settings.r_bins_edges_dry,
                        dry=True,
                    ),
                ]
            )
        if settings.precip:
            self.products.extend(
                [
                    PySDM_products.CollisionRatePerGridbox(
                        name="collision_rate",
                    ),
                    PySDM_products.CollisionRateDeficitPerGridbox(
                        name="collision_deficit",
                    ),
                    PySDM_products.CoalescenceRatePerGridbox(
                        name="coalescence_rate",
                    ),
                    PySDM_products.SurfacePrecipitation(),
                ]
            )
        self.particulator = self.builder.build(
            attributes=self.attributes, products=tuple(self.products)
        )

        self.output_attributes = {
            "cell origin": [],
            "position in cell": [],
            "radius": [],
            "multiplicity": [],
        }
        self.output_products = {}
        for k, v in self.particulator.products.items():
            if len(v.shape) == 0:
                self.output_products[k] = np.zeros(self.nt + 1)
            elif len(v.shape) == 1:
                self.output_products[k] = np.zeros((self.mesh.grid[-1], self.nt + 1))
            elif len(v.shape) == 2:
                number_of_time_sections = len(self.save_spec_and_attr_times)
                self.output_products[k] = np.zeros(
                    (self.mesh.grid[-1], self.number_of_bins, number_of_time_sections)
                )

    @staticmethod
    def add_collision_dynamic(builder, settings, _):
        builder.add_dynamic(
            Coalescence(
                collision_kernel=settings.collision_kernel,
                adaptive=settings.coalescence_adaptive,
            )
        )

    def save_scalar(self, step):
        for k, v in self.particulator.products.items():
            if len(v.shape) > 1:
                continue
            if len(v.shape) == 1:
                self.output_products[k][:, step] = v.get()
            else:
                self.output_products[k][step] = v.get()

    def save_spectrum(self, index):
        for k, v in self.particulator.products.items():
            if len(v.shape) == 2:
                self.output_products[k][:, :, index] = v.get()

    def save_attributes(self):
        for k, v in self.output_attributes.items():
            v.append(self.particulator.attributes[k].to_ndarray())

    def save(self, step):
        self.save_scalar(step)
        time = step * self.particulator.dt
        if len(self.save_spec_and_attr_times) > 0 and (
            np.min(
                np.abs(
                    np.ones_like(self.save_spec_and_attr_times) * time
                    - np.array(self.save_spec_and_attr_times)
                )
            )
            < 0.1
        ):
            save_index = np.argmin(
                np.abs(
                    np.ones_like(self.save_spec_and_attr_times) * time
                    - np.array(self.save_spec_and_attr_times)
                )
            )
            self.save_spectrum(save_index)
            self.save_attributes()

    def run(self):
        mesh = self.particulator.mesh

        assert "t" not in self.output_products and "z" not in self.output_products
        self.output_products["t"] = np.linspace(
            0, self.nt * self.particulator.dt, self.nt + 1, endpoint=True
        )
        self.output_products["z"] = np.linspace(
            self.z0 + mesh.dz / 2,
            self.z0 + (mesh.grid[-1] - 1 / 2) * mesh.dz,
            mesh.grid[-1],
            endpoint=True,
        )

        self.save(0)
        for step in range(self.nt):
            self.mpdata.update_advector_field()
            if "Displacement" in self.particulator.dynamics:
                self.particulator.dynamics["Displacement"].upload_courant_field(
                    (self.mpdata.advector / self.g_factor_vec,)
                )
            self.particulator.run(steps=1)
            self.save(step + 1)

        Outputs = namedtuple("Outputs", "products attributes")
        output_results = Outputs(self.output_products, self.output_attributes)
        return output_results
