from collections import namedtuple

import numpy as np
from PySDM_examples.Shipway_and_Hill_2012.mpdata_1d import MPDATA_1D

import PySDM.products as PySDM_products
from PySDM import Builder
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

from typing import Iterable, Optional

import numpy as np
from numdifftools import Derivative
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from PySDM import Formulae
from PySDM.dynamics import condensation
from PySDM.initialisation import spectra
from PySDM.physics import si
from PySDM.dynamics.collisions.collision_kernels import Geometric


class Settings:
    def __dir__(self) -> Iterable[str]:
        return (
            "n_sd_per_gridbox",
            "p0",
            "kappa",
            "rho_times_w_1",
            "particles_per_volume_STP",
            "dt",
            "dz",
            "precip",
            "z_max",
            "t_max",
            "cloud_water_radius_range",
            "rain_water_radius_range",
            "r_bins_edges_dry",
            "r_bins_edges",
        )

    def __init__(
        self,
        *,
        n_sd_per_gridbox: int,
        p0: float = 1007 * si.hPa,  # as used in Olesik et al. 2022 (GMD)
        kappa: float = 1,
        rho_times_w_1: float = 2 * si.m / si.s * si.kg / si.m**3,
        particles_per_volume_STP: int = 50 / si.cm**3,
        dt: float = 1 * si.s,
        dz: float = 25 * si.m,
        z_max: float = 3000 * si.m,
        z_part: Optional[tuple] = None,
        t_max: float = 60 * si.minutes,
        precip: bool = True,
        enable_condensation: bool = True,
        formulae: Formulae = None,
        save_spec_and_attr_times=(),
        collision_kernel=None,
        alpha = 0,
    ):
        self.formulae = formulae or Formulae()
        self.n_sd_per_gridbox = n_sd_per_gridbox
        self.p0 = p0
        self.kappa = kappa
        self.rho_times_w_1 = rho_times_w_1
        self.particles_per_volume_STP = particles_per_volume_STP
        self.dt = dt
        self.dz = dz
        self.precip = precip
        self.enable_condensation = enable_condensation
        self.z_part = z_part
        self.z_max = z_max
        self.t_max = t_max
        self.collision_kernel = collision_kernel or Geometric(collection_efficiency=1)
        self.alpha = alpha

        t_1 = 600 * si.s
        self.rho_times_w = lambda t: (
            rho_times_w_1 * np.sin(np.pi * t / t_1) if t < t_1 else 0
        )
        apprx_w1 = rho_times_w_1 / self.formulae.constants.rho_STP
        self.particle_reservoir_depth = (
            (2 * apprx_w1 * t_1 / np.pi) // self.dz + 1
        ) * self.dz

        self.wet_radius_spectrum_per_mass_of_dry_air = spectra.Lognormal(
            norm_factor=particles_per_volume_STP / self.formulae.constants.rho_STP,
            m_mode=0.08 / 2 * si.um,
            s_geom=1.4,
        )

        self._th = interp1d(
            (0.0 * si.m, 740.0 * si.m, 3260.00 * si.m),
            (297.9 * si.K, 297.9 * si.K, 312.66 * si.K),
            fill_value="extrapolate",
        )

        self.water_vapour_mixing_ratio = interp1d(
            (-max(self.particle_reservoir_depth, 1), 0, 740, 3260),
            (0.015, 0.015, 0.0138, 0.0024),
            fill_value="extrapolate",
        )

        self.thd = (
            lambda z_above_reservoir: self.formulae.state_variable_triplet.th_dry(
                self._th(z_above_reservoir),
                self.water_vapour_mixing_ratio(z_above_reservoir),
            )
        )

        self.rhod0 = self.formulae.state_variable_triplet.rho_d(
            p=p0,
            water_vapour_mixing_ratio=self.water_vapour_mixing_ratio(0 * si.m),
            theta_std=self._th(0 * si.m),
        )

        def drhod_dz(z_above_reservoir, rhod):
            if z_above_reservoir < 0:
                return 0
            water_vapour_mixing_ratio = self.water_vapour_mixing_ratio(
                z_above_reservoir
            )
            d_water_vapour_mixing_ratio__dz = 0#Derivative(
            #     self.water_vapour_mixing_ratio
            # )(z_above_reservoir)
            T = self.formulae.state_variable_triplet.T(
                rhod[0], self.thd(z_above_reservoir)
            )
            p = self.formulae.state_variable_triplet.p(
                rhod[0], T, water_vapour_mixing_ratio
            )
            lv = self.formulae.latent_heat_vapourisation.lv(T)
            return self.formulae.hydrostatics.drho_dz(
                p, T, water_vapour_mixing_ratio, lv
            ) / (
                1 + water_vapour_mixing_ratio
            ) - rhod * d_water_vapour_mixing_ratio__dz / (
                1 + water_vapour_mixing_ratio
            )

        z_span = (-self.particle_reservoir_depth, self.z_max)
        z_points = np.linspace(*z_span, 2 * self.nz + 1)
        rhod_solution = solve_ivp(
            fun=drhod_dz,
            t_span=z_span,
            y0=np.asarray((self.rhod0,)),
            t_eval=z_points,
            max_step=dz / 2,
        )
        assert rhod_solution.success
        self.rhod = interp1d(z_points, rhod_solution.y[0])

        self.mpdata_settings = {"n_iters": 3, "iga": True, "fct": True, "tot": True}
        self.condensation_rtol_x = condensation.DEFAULTS.rtol_x
        self.condensation_rtol_thd = condensation.DEFAULTS.rtol_thd
        self.condensation_adaptive = True
        self.condensation_update_thd = False
        self.coalescence_adaptive = True

        self.number_of_bins = 100
        self.r_bins_edges_dry = np.logspace(
            np.log10(0.001 * si.um),
            np.log10(1 * si.um),
            self.number_of_bins + 1,
            endpoint=True,
        )
        self.r_bins_edges = np.logspace(
            np.log10(0.001 * si.um),
            np.log10(10 * si.mm),
            self.number_of_bins + 1,
            endpoint=True,
        )
        self.cloud_water_radius_range = [2 * si.um, 50 * si.um] # edited from 1um in 2012 example
        self.cloud_water_radius_range_igel = [2 * si.um, 25 * si.um]
        self.rain_water_radius_range = [50 * si.um, np.inf * si.um]
        self.rain_water_radius_range_igel = [25 * si.um, np.inf * si.um]
        self.save_spec_and_attr_times = save_spec_and_attr_times

    @property
    def n_sd(self):
        return self.nz * self.n_sd_per_gridbox

    @property
    def nz(self):
        assert (
            self.particle_reservoir_depth / self.dz
            == self.particle_reservoir_depth // self.dz
        )
        nz = (self.z_max + self.particle_reservoir_depth) / self.dz
        assert nz == int(nz)
        return int(nz)

    @property
    def nt(self):
        nt = self.t_max / self.dt
        assert nt == int(nt)
        return int(nt)

class Simulation:
    def __init__(self, settings, backend):
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

        env = Kinematic1D(
            dt=settings.dt,
            mesh=self.mesh,
            thd_of_z=settings.thd,
            rhod_of_z=settings.rhod,
            z0=-settings.particle_reservoir_depth,
        )

        def zZ_to_z_above_reservoir(zZ):
            z_above_reservoir = zZ * (settings.nz * settings.dz) + self.z0
            return z_above_reservoir

        mpdata = MPDATA_1D(
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
            backend=backend,
            environment=env,
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
        self.builder.add_dynamic(EulerianAdvection(mpdata))

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
        self.attributes = self.builder.particulator.environment.init_attributes(
            spatial_discretisation=spatial_sampling.Pseudorandom(),
            spectral_discretisation=spectral_sampling.AlphaSampling(
                spectrum=settings.wet_radius_spectrum_per_mass_of_dry_air,
                alpha=settings.alpha,
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
            PySDM_products.LiquidWaterContent(
                name="LWC",
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
                # radius_range=settings.cloud_water_radius_range
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
                    PySDM_products.PeakSaturation(),
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
            mpdata = self.particulator.dynamics["EulerianAdvection"].solvers
            mpdata.update_advector_field()
            if "Displacement" in self.particulator.dynamics:
                self.particulator.dynamics["Displacement"].upload_courant_field(
                    (mpdata.advector / self.g_factor_vec,)
                )
            self.particulator.run(steps=1)
            self.save(step + 1)

        Outputs = namedtuple("Outputs", "products attributes")
        output_results = Outputs(self.output_products, self.output_attributes)
        return output_results
