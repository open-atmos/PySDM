from typing import Any, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib import animation
from open_atmos_jupyter_utils import show_plot
from PySDM_examples.Bhalla_2023.logging_observers import Progress
from PySDM_examples.Bhalla_2023.plot_terminal_velocity_approx import get_approx
from PySDM_examples.Bhalla_2023.settings_0D import Settings
from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumColors

from PySDM.backends import CPU, GPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence, RelaxedVelocity
from PySDM.environments import Box
from PySDM.impl.wall_timer import WallTimer
from PySDM.initialisation import init_fall_momenta
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si
from PySDM.products import (
    ParticleVolumeVersusRadiusLogarithmSpectrum,
    RadiusBinnedNumberAveragedFallVelocity,
    RadiusBinnedNumberAveragedTerminalVelocity,
    WallTime,
)


class Simulation:
    """
    Based on PySDM_examples.Shima_et_al_2009.example.run
    """

    def __init__(self, settings: Settings, backend: Union[Type[CPU], Type[GPU]] = CPU):
        self.settings = settings

        self.builder = Builder(
            n_sd=self.settings.n_sd, backend=backend(formulae=self.settings.formulae)
        )
        self.builder.set_environment(Box(dv=self.settings.dv, dt=self.settings.dt))

        ##############
        # ATTRIBUTES #
        ##############
        attributes = {}

        sampling = ConstantMultiplicity(self.settings.spectrum)
        attributes["volume"], attributes["n"] = sampling.sample(self.settings.n_sd)

        self.builder.request_attribute("fall velocity")
        attributes["fall momentum"] = init_fall_momenta(
            attributes["volume"], self.builder.formulae.constants.rho_w
        )

        self.builder.request_attribute("radius")

        ############
        # DYNAMICS #
        ############

        relaxed_velocity = RelaxedVelocity(tau=self.settings.tau)
        self.builder.add_dynamic(relaxed_velocity)

        coalescence = Coalescence(
            collision_kernel=self.settings.kernel, adaptive=self.settings.adaptive
        )
        self.builder.add_dynamic(coalescence)

        ############
        # PRODUCTS #
        ############

        products = [
            ParticleVolumeVersusRadiusLogarithmSpectrum(
                self.settings.radius_bins_edges, name="dv/dlnr"
            ),
            RadiusBinnedNumberAveragedTerminalVelocity(
                self.settings.radius_bins_edges, name="terminal_vel"
            ),
            RadiusBinnedNumberAveragedFallVelocity(
                self.settings.radius_bins_edges, name="fall_vel"
            ),
            WallTime(),
        ]

        self.particulator = self.builder.build(attributes, tuple(products))

        #############
        # OBSERVERS #
        #############

        self.particulator.observers.append(Progress(self.settings.output_steps[-1]))

        ############
        # OUTPUT   #
        ############

        self.dt = self.settings.dt
        self.num_steps = int(self.settings.max_t / self.dt)
        self.done: bool = False
        self._output_products: dict[str, list[Any]] = {
            "dm/dlnr": [None for _ in range(self.num_steps + 1)],
            "terminal_vel": [None for _ in range(self.num_steps + 1)],
            "fall_vel": [None for _ in range(self.num_steps + 1)],
        }
        self._output_attributes: dict[str, list[Any]] = {
            a: [None for _ in range(self.num_steps + 1)]
            for a in self.particulator.attributes.keys()
        }
        self._exec_time: float = -1

    def save(self, index):
        self._output_products["dm/dlnr"][index] = self.particulator.products[
            "dv/dlnr"
        ].get()[0]
        self._output_products["dm/dlnr"][index][:] *= self.settings.rho

        self._output_products["terminal_vel"][index] = self.particulator.products[
            "terminal_vel"
        ].get()

        self._output_products["fall_vel"][index] = self.particulator.products[
            "fall_vel"
        ].get()

        for a in self._output_attributes:
            self._output_attributes[a][index] = self.particulator.attributes[
                a
            ].to_ndarray()

    def run(self):
        assert not self.done

        self.particulator.products["wall time"].reset()

        self.save(0)
        for step in range(1, self.num_steps + 1):
            self.particulator.run(1)
            self.save(step)

        self._exec_time = self.particulator.products["wall time"].get()
        self.done = True

    def get_plt_name(self, plot_var: str) -> str:
        res = f"0D {plot_var} relax_vel={self.settings.evaluate_relaxed_velocity}"
        if self.settings.evaluate_relaxed_velocity:
            res += f" tau={self.settings.tau}s"
        res += f" v_t={self.builder.get_attribute('terminal velocity').approximation.__class__.__name__}"
        return res

    def _plot_vs_lnr_single(
        self,
        index: int,
        product_name: str,
        y_scale: float,
        colors: SpectrumColors,
        set_to=None,
    ):
        if set_to is not None:
            set_to.set_data(
                self.settings.radius_bins_edges[:-1] * si.metres / si.micrometres,
                self._output_products[product_name][index] * y_scale,
            )
            set_to.set_label(f"t = {index*self.dt}s")
            set_to.set_color(colors(index / self.num_steps))
            return set_to

        else:
            return plt.step(
                self.settings.radius_bins_edges[:-1] * si.metres / si.micrometres,
                self._output_products[product_name][index] * y_scale,
                where="post",
                label=f"t = {index*self.dt}s",
                color=colors(index / self.num_steps),
            )

    def plot_vs_lnr_animation(
        self,
        product_name: str,
        y_scale: float = 1,
        y_label: Optional[str] = None,
        num_fixed: int = 4,
        show=True,
        speedup=40,
    ):
        assert self.done

        plt_colors = SpectrumColors()

        fig = plt.figure()
        (graph,) = self._plot_vs_lnr_single(0, product_name, y_scale, plt_colors)

        stamp_steps = np.linspace(0, self.num_steps, num_fixed)
        for i in stamp_steps:
            (new_graph,) = self._plot_vs_lnr_single(
                int(i), product_name, y_scale, plt_colors
            )
            new_graph.set_alpha(0.5)

        def animate(i):
            update_list = [
                self._plot_vs_lnr_single(
                    i * speedup, product_name, y_scale, plt_colors, set_to=graph
                ),
                plt.legend(),
            ]

            return update_list

        anim = animation.FuncAnimation(
            fig, animate, frames=self.num_steps // speedup, interval=50
        )

        plt_name = self.get_plt_name(product_name.replace("/", "_"))

        plt.xscale("log", base=10)

        plt.title(plt_name)
        plt.xlabel("particle radius [µm]")
        plt.ylabel(y_label)

        output_nums = self._output_products[product_name]
        plt.xlim(
            (
                np.min(self.settings.radius_bins_edges) * si.metres / si.micrometres,
                np.max(self.settings.radius_bins_edges) * si.metres / si.micrometres,
            )
        )
        plt.ylim((0, np.max(output_nums) * y_scale))

        anim.save(f"{plt_name}.gif", fps=30)

        if show:
            try:
                assert get_ipython().__class__.__name__ == "ZMQInteractiveShell"
                display(HTML(anim.to_html5_video()))
                plt.close()
            except:
                plt.show()

    def plot_sd_velocity_animation(self, show=True, speedup=40):
        """
        Create an animation similar to the plot ln vs r animation
        except it plots the velocity attributes at each index
        """
        assert self.done

        def plot_at_index(i, set_to=None):
            if set_to is not None:
                set_to.set_data(
                    self._output_attributes["radius"][i] * si.metres / si.micrometres,
                    self._output_attributes["fall velocity"][i],
                )
                set_to.set_label(f"t = {i*self.dt}s")
                return set_to

            else:
                return plt.plot(
                    self._output_attributes["radius"][i] * si.metres / si.micrometres,
                    self._output_attributes["fall velocity"][i],
                    "ro",
                )

        LOG_LOWER_BOUND = -6
        LOG_UPPER_BOUND = -2

        radii_arr = np.logspace(LOG_LOWER_BOUND, LOG_UPPER_BOUND, 1000)
        terminal_vel_arr = get_approx(radii_arr, particulator=self.particulator)

        fig = plt.figure()
        (graph,) = plot_at_index(0)

        plt.plot(
            radii_arr * si.metres / si.micrometres,
            terminal_vel_arr,
            "b-",
            label="Rogers Yau",
            alpha=0.5,
            linewidth=3,
        )

        def animate(i):
            return [plot_at_index(i * speedup, set_to=graph), plt.legend()]

        anim = animation.FuncAnimation(
            fig, animate, frames=self.num_steps // speedup, interval=50
        )

        plt_name = self.get_plt_name("sd_velocity")

        plt.xscale("log", base=10)

        plt.title(plt_name)
        plt.xlabel("particle radius [µm]")
        plt.ylabel("velocity [m/s]")

        anim.save(f"{plt_name}.gif", fps=30)

        if show:
            try:
                assert get_ipython().__class__.__name__ == "ZMQInteractiveShell"
                display(HTML(anim.to_html5_video()))
                plt.close()
            except:
                plt.show()
