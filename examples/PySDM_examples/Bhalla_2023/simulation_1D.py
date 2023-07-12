from collections import namedtuple
from IPython.display import display, HTML
from PySDM.products.collision import CollisionRatePerGridbox
from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumColors
from PySDM_examples.Shipway_and_Hill_2012.plot import plot
from PySDM_examples.Shipway_and_Hill_2012.simulation import Simulation as Simulation_Shipway
from matplotlib import animation

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union

from PySDM_examples.Shipway_and_Hill_2012.mpdata_1d import MPDATA_1D
from open_atmos_jupyter_utils import show_plot
from PySDM import Builder
from PySDM.dynamics.relaxed_velocity import RelaxedVelocity
from PySDM.initialisation import init_fall_momenta

import PySDM.products as PySDM_products
from PySDM.backends import CPU
from PySDM.dynamics import (
    AmbientThermodynamics,
    Coalescence,
    Condensation,
    Displacement,
    EulerianAdvection,
)

from PySDM.dynamics.collisions.collision_kernels import Geometric
from PySDM.environments.kinematic_1d import Kinematic1D
from PySDM.impl.mesh import Mesh
from PySDM.initialisation.sampling import spatial_sampling, spectral_sampling

from PySDM_examples.Bhalla_2023.logging_observers import Progress
from PySDM_examples.Bhalla_2023.settings_1D import Settings

from PySDM.physics import si


class Simulation(Simulation_Shipway):
    """
    Based on PySDM_examples.Shipway_and_Hill_2012.simulation.Simulation
    """

    def __init__(self, settings: Settings, backend=CPU):
        self.settings = settings
        self.nt = settings.nt
        self.z0 = -settings.particle_reservoir_depth
        self.save_spec_and_attr_times = settings.save_spec_and_attr_times
        self.number_of_bins = settings.number_of_bins

        self.particulator = None
        self.output_attributes = None
        self.output_products = None

        self.exec_time = -1
        self.done = False

        self.builder = Builder(
            n_sd=settings.n_sd, backend=backend(formulae=settings.formulae)
        )
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
            advector_of_t=lambda t: settings.rho_times_w(
                t) * settings.dt / settings.dz,
            advectee_of_zZ_at_t0=lambda zZ: settings.qv(
                zZ_to_z_above_reservoir(zZ)),
            g_factor_of_zZ=lambda zZ: settings.rhod(
                zZ_to_z_above_reservoir(zZ)),
        )

        _extra_nz = settings.particle_reservoir_depth // settings.dz
        _z_vec = settings.dz * np.linspace(
            -_extra_nz, settings.nz - _extra_nz, settings.nz + 1
        )
        self.g_factor_vec = settings.rhod(_z_vec)

        self.builder.set_environment(self.env)
        self.builder.add_dynamic(AmbientThermodynamics())
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
            relax_velocity=settings.evaluate_relaxed_velocity
        )
        self.builder.add_dynamic(displacement)
        self.attributes = self.env.init_attributes(
            spatial_discretisation=spatial_sampling.Pseudorandom(),
            spectral_discretisation=spectral_sampling.ConstantMultiplicity(
                spectrum=settings.wet_radius_spectrum_per_mass_of_dry_air
            ),
            kappa=settings.kappa,
        )

        if self.settings.evaluate_relaxed_velocity:
            relaxed_velocity = RelaxedVelocity(tau=self.settings.tau)
            self.builder.add_dynamic(relaxed_velocity)

            self.attributes["fall momentum"] = init_fall_momenta(
                self.attributes["volume"], self.builder.formulae.constants.rho_w)

            self.builder.request_attribute("fall velocity")

        self.products += [
            # PySDM_products.AmbientRelativeHumidity(name="RH", unit="%"),
            # PySDM_products.AmbientPressure(name="p"),
            # PySDM_products.AmbientTemperature(name="T"),
            # PySDM_products.AmbientWaterVapourMixingRatio(name="qv"),
            PySDM_products.WaterMixingRatio(
                name="qc", unit="g/kg", radius_range=settings.cloud_water_radius_range
            ),
            PySDM_products.WaterMixingRatio(
                name="qr", unit="g/kg", radius_range=settings.rain_water_radius_range
            ),
            # PySDM_products.AmbientDryAirDensity(name="rhod"),
            # PySDM_products.AmbientDryAirPotentialTemperature(name="thd"),
            # PySDM_products.ParticleSizeSpectrumPerVolume(
            #     name="dry spectrum",
            #     radius_bins_edges=settings.r_bins_edges_dry,
            #     dry=True,
            # ),
            # PySDM_products.ParticleSizeSpectrumPerVolume(
            #     name="wet spectrum", radius_bins_edges=settings.r_bins_edges
            # ),
            # PySDM_products.ParticleConcentration(
            #     name="nc", radius_range=settings.cloud_water_radius_range
            # ),
            # PySDM_products.ParticleConcentration(
            #     name="nr", radius_range=settings.rain_water_radius_range
            # ),
            # PySDM_products.ParticleConcentration(
            #     name="na", radius_range=(0, settings.cloud_water_radius_range[0])
            # ),
            # PySDM_products.MeanRadius(),
            # PySDM_products.RipeningRate(name="ripening"),
            # PySDM_products.ActivatingRate(name="activating"),
            # PySDM_products.DeactivatingRate(name="deactivating"),
            # PySDM_products.EffectiveRadius(
            #     radius_range=settings.cloud_water_radius_range
            # ),
            # PySDM_products.PeakSupersaturation(unit="%"),
            # PySDM_products.SuperDropletCountPerGridbox(),
            # PySDM_products.AveragedTerminalVelocity(
            #     name="rain averaged terminal velocity",
            #     radius_range=settings.rain_water_radius_range,
            # ),
            PySDM_products.WallTime()
        ]
        if settings.precip:
            # self.products.append(
            #     PySDM_products.CollisionRatePerGridbox(
            #         name="collision_rate",
            #     ),
            # )
            # self.products.append(
            #     PySDM_products.CollisionRateDeficitPerGridbox(
            #         name="collision_deficit",
            #     ),
            # )
            # self.products.append(
            #     PySDM_products.CoalescenceRatePerGridbox(
            #         name="coalescence_rate",
            #     ),
            # )
            pass

        self.products.append(PySDM_products.NumberSizeSpectrum(
            self.settings.r_bins_edges, name="number concentration"))

        self.products.append(
            PySDM_products.RadiusBinnedNumberAveragedTerminalVelocity(
                self.settings.r_bins_edges, name="terminal_vel"
            )
        )

        self.products.append(
            PySDM_products.ParticleVolumeVersusRadiusLogarithmSpectrum(
                self.settings.r_bins_edges, name="dv/dlnr"
            )
        )

        if self.settings.evaluate_relaxed_velocity:
            self.products.append(PySDM_products.RadiusBinnedNumberAveragedFallVelocity(
                self.settings.r_bins_edges, name="fall_vel"
            ))

        self.particulator = self.builder.build(
            attributes=self.attributes, products=self.products
        )

        self.particulator.observers.append(Progress(self.nt))

        self.output_attributes = {
            "cell origin": [],
            "position in cell": [],
            "radius": [],
            "n": []
        }

        self.output_attributes["terminal velocity"] = []
        if self.settings.evaluate_relaxed_velocity:
            self.output_attributes["fall velocity"] = []

        self.output_products = {}
        for k, v in self.particulator.products.items():
            if len(v.shape) == 1:
                self.output_products[k] = np.zeros(
                    (self.mesh.grid[-1], self.nt + 1))
            elif len(v.shape) == 2:
                number_of_time_sections = len(self.save_spec_and_attr_times)
                self.output_products[k] = np.zeros(
                    (self.mesh.grid[-1], self.number_of_bins,
                     number_of_time_sections)
                )

    @staticmethod
    def add_collision_dynamic(builder, settings, _):
        builder.add_dynamic(
            Coalescence(
                collision_kernel=Geometric(
                    collection_efficiency=1, relax_velocity=settings.evaluate_relaxed_velocity),
                adaptive=settings.coalescence_adaptive,
            )
        )

    def save_scalar(self, step):
        for k, v in self.particulator.products.items():
            if len(v.shape) > 1:
                continue
            elif k == "wall time":
                continue
            self.output_products[k][:, step] = v.get()

    def save_spectrum(self, index):
        for k, v in self.particulator.products.items():
            if len(v.shape) == 2:
                self.output_products[k][:, :, index] = v.get()

    def run(self):
        assert not self.done

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
        self.particulator.products["wall time"].reset()
        for step in range(self.nt):
            self.mpdata.update_advector_field()
            if "Displacement" in self.particulator.dynamics:
                self.particulator.dynamics["Displacement"].upload_courant_field(
                    (self.mpdata.advector / self.g_factor_vec,)
                )
            self.particulator.run(steps=1)
            self.save(step + 1)
        self.exec_time = self.particulator.products["wall time"].get()

        Outputs = namedtuple("Outputs", "products attributes")
        output_results = Outputs(self.output_products, self.output_attributes)

        self.done = True

        return output_results

    def get_plt_name(self, plot_var: str)->str:
        res = f"1D {plot_var} relax_vel={self.settings.evaluate_relaxed_velocity}"
        if self.settings.evaluate_relaxed_velocity:
            res += f" tau={self.settings.tau}s"
        res += f" v_t={self.builder.get_attribute('terminal velocity').approximation.__class__.__name__}"
        return res

    def get_total_spectrum(self, product_name, index):
        """
        either takes a weighted average w.r.t. "number concentration" product
        or takes a sum
        """
        prod = self.output_products[product_name][:, :, index]
        if product_name == "terminal_vel" or product_name == "fall_vel":
            num = self.output_products["number concentration"][:, :, index]
            weighted_sum = np.sum(prod*num, axis=0)
            total_num = np.sum(num, axis=0)
            return np.divide(weighted_sum, total_num, out=np.zeros_like(weighted_sum), where=total_num!=0)
        elif product_name == "dv/dlnr":
            return np.sum(prod, axis=0)
        else:
            raise ValueError("no total spectrum behavior specified")


    def _plot_vs_lnr_single(self, index: int, product_name: str, y_scale: float, colors: SpectrumColors, set_to=None):
        X = np.linspace(1, 100, 50)
        Y = X*0.05
        if set_to is not None:
            set_to.set_data(self.settings.r_bins_edges[:-
                                                       1] * si.metres / si.micrometres,
                            self.get_total_spectrum(product_name, index) * y_scale)
            set_to.set_label(f"t = {self.settings.save_spec_and_attr_times[index]}s")
            set_to.set_color(colors(
                self.settings.save_spec_and_attr_times[index] /
                (self.settings.t_max)
            ))
            return set_to

        else:
            return plt.step(
                self.settings.r_bins_edges[:-
                                           1] * si.metres / si.micrometres,
                self.get_total_spectrum(product_name, index) * y_scale,
                where="post",
                label=f"t = {self.settings.save_spec_and_attr_times[index]}s",
                color=colors(
                    self.settings.save_spec_and_attr_times[index] /
                    (self.settings.t_max)
                ),
            )

    def plot_vs_lnr_animation(self, product_name: str, y_scale: float = 1, y_label: Optional[str] = None, num_fixed: int = 4, show=True):
        assert self.done

        plt_colors = SpectrumColors()

        fig = plt.figure()
        graph, = self._plot_vs_lnr_single(0, product_name, y_scale, plt_colors)

        stamp_steps = np.linspace(
            0, len(self.settings.save_spec_and_attr_times)-1, num_fixed)
        for i in stamp_steps:
            new_graph, = self._plot_vs_lnr_single(
                int(i), product_name, y_scale, plt_colors)
            new_graph.set_alpha(0.5)

        def animate(i):
            update_list = [self._plot_vs_lnr_single(
                i, product_name, y_scale, plt_colors, set_to=graph), plt.legend()]

            return update_list

        anim = animation.FuncAnimation(fig, animate,
                                       frames=len(self.settings.save_spec_and_attr_times), interval=50)

        plt_name = self.get_plt_name(product_name.replace("/", "_"))

        plt.xscale("log", base=10)

        plt.title(plt_name)
        plt.xlabel("particle radius [µm]")
        plt.ylabel(y_label)

        # output_nums = [i for i in self.get_total_spectrum(product_name, slice(None))]
        # plt.xlim((np.min(self.settings.r_bins_edges) * si.metres / si.micrometres,
        #           np.max(self.settings.r_bins_edges) * si.metres / si.micrometres))
        # plt.ylim((0, np.max(output_nums) * y_scale))

        anim.save(f"{plt_name}.gif", fps=30)

        if show:
            try:
                assert get_ipython().__class__.__name__ == "ZMQInteractiveShell"
                display(HTML(anim.to_html5_video()))
                plt.close()
            except:
                plt.show()


def generate_plots(evaluate_relaxed_velocity=True, tau=100*si.seconds):
    settings = Settings(times_to_save=np.array(
        [0, 1000, 2000, 3000]), evaluate_relaxed_velocity=evaluate_relaxed_velocity, tau=tau)
    simulation = Simulation(settings)
    results = simulation.run()
    products = results.products
    plot(var="qc", qlabel="$q_c$ [g/kg]", fname=simulation.get_plt_name("q_c")+".pdf",
         output=products)
    plot(var='qr', qlabel='$q_r$ [g/kg]', fname=simulation.get_plt_name("q_r")+".pdf",
             output=products)


if __name__ == "__main__":
    generate_plots(evaluate_relaxed_velocity=False)
    generate_plots(evaluate_relaxed_velocity=True, tau=1*si.seconds)
    generate_plots(evaluate_relaxed_velocity=True, tau=10*si.seconds)
    generate_plots(evaluate_relaxed_velocity=True, tau=100*si.seconds)
