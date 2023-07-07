from typing import Any, Optional, Type, Union
import numpy as np
from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumColors
import matplotlib.pyplot as plt
from matplotlib import animation
from PySDM.backends import CPU, GPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence, RelaxedVelocity
from PySDM.environments import Box
from PySDM.impl.wall_timer import WallTimer
from PySDM.initialisation import init_fall_momenta
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import ParticleVolumeVersusRadiusLogarithmSpectrum, RadiusBinnedNumberAveragedTerminalVelocity, RadiusBinnedNumberAveragedFallVelocity, WallTime
from PySDM.physics import si
from open_atmos_jupyter_utils import show_plot
from PySDM_examples.Bhalla_2023.settings_0D import Settings
from PySDM_examples.Bhalla_2023.logging_observers import Progress
from IPython.display import display, HTML


class Simulation:
    """
    Based on PySDM_examples.Shima_et_al_2009.example.run
    """

    def __init__(self, settings: Settings, backend: Union[Type[CPU], Type[GPU]] = CPU):
        self.settings = settings

        self.builder = Builder(n_sd=self.settings.n_sd, backend=backend(
            formulae=self.settings.formulae))
        self.builder.set_environment(
            Box(dv=self.settings.dv, dt=self.settings.dt))

        attributes = {}

        sampling = ConstantMultiplicity(self.settings.spectrum)
        attributes["volume"], attributes["n"] = sampling.sample(
            self.settings.n_sd)

        coalescence = Coalescence(
            collision_kernel=self.settings.kernel, adaptive=self.settings.adaptive
        )
        self.builder.add_dynamic(coalescence)

        if self.settings.evaluate_relaxed_velocity:
            relaxed_velocity = RelaxedVelocity(tau=self.settings.tau)
            self.builder.add_dynamic(relaxed_velocity)

            attributes["fall momentum"] = init_fall_momenta(
                attributes["volume"], self.builder.formulae.constants.rho_w)

            self.builder.request_attribute("fall velocity")

        products = [
            ParticleVolumeVersusRadiusLogarithmSpectrum(
                self.settings.radius_bins_edges, name="dv/dlnr"
            ),
            RadiusBinnedNumberAveragedTerminalVelocity(
                self.settings.radius_bins_edges, name="terminal_vel"
            ),
            WallTime(),
        ]

        if self.settings.evaluate_relaxed_velocity:
            products.append(RadiusBinnedNumberAveragedFallVelocity(
                self.settings.radius_bins_edges, name="fall_vel"
            ))

        self.particulator = self.builder.build(attributes, tuple(products))

        self.particulator.observers.append(
            Progress(self.settings.output_steps[-1]))

        # self.particulator.observers.append(WarnVelocityDiff(self.particulator))

        self.done: bool = False
        self._output: list[dict[str, Any]] = []
        self._exec_time: float = -1

    def run(self):
        assert not self.done

        self._output = []
        self.particulator.products["wall time"].reset()
        for step in self.settings.output_steps:
            self.particulator.run(step - self.particulator.n_steps)

            output_vals: dict[str, Any] = {}

            output_vals["dm/dlnr"] = self.particulator.products["dv/dlnr"].get()[0]
            output_vals["dm/dlnr"][:] *= self.settings.rho

            output_vals["terminal_vel"] = self.particulator.products["terminal_vel"].get()

            if self.settings.evaluate_relaxed_velocity:
                output_vals["fall_vel"] = self.particulator.products["fall_vel"].get()

            self._output.append(output_vals)

        self._exec_time = self.particulator.products["wall time"].get()
        self.done = True

    def get_plt_name(self, plot_var: str)->str:
        res = f"0D {plot_var} relax_vel={self.settings.evaluate_relaxed_velocity}"
        if self.settings.evaluate_relaxed_velocity:
            res += f" tau={self.settings.tau}s"
        res += f" v_t={self.builder.get_attribute('terminal velocity').approximation.__class__.__name__}"
        return res

    def _plot_vs_lnr_single(self, index: int, product_name: str, y_scale: float, colors: SpectrumColors, set_to=None):
        X = np.linspace(1, 100, 50)
        Y = X*0.05
        if set_to is not None:
            set_to.set_data(self.settings.radius_bins_edges[:-
                                                            1] * si.metres / si.micrometres,
                            self._output[index][product_name] * y_scale)
            set_to.set_label(f"t = {self.settings.steps[index]}s")
            set_to.set_color(colors(
                self.settings.steps[index] /
                (self.settings.output_steps[-1] * self.settings.dt)
            ))
            return set_to

        else:
            return plt.step(
                self.settings.radius_bins_edges[:-
                                                1] * si.metres / si.micrometres,
                self._output[index][product_name] * y_scale,
                where="post",
                label=f"t = {self.settings.steps[index]}s",
                color=colors(
                    self.settings.steps[index] /
                    (self.settings.output_steps[-1] * self.settings.dt)
                ),
            )

    def plot_vs_lnr(self, product_name: str, y_scale: float = 1, y_label: Optional[str] = None, num_plots: int = 4):
        assert self.done

        y_label = y_label or product_name

        plt_colors = SpectrumColors()

        steps = np.linspace(0, len(self.settings.steps)-1, num_plots)
        for i in steps:
            self._plot_vs_lnr_single(
                int(i), product_name, y_scale, plt_colors)

        plt_name = self.get_plt_name(product_name.replace("/", "_"))

        plt.xscale("log", base=10)

        plt.title(plt_name)
        plt.xlabel("particle radius [µm]")
        plt.ylabel(y_label)
        plt.legend()

        show_plot(filename=f"{plt_name}.pdf")

    def plot_vs_lnr_animation(self, product_name: str, y_scale: float = 1, y_label: Optional[str] = None, num_fixed: int = 4, show=True):
        assert self.done

        plt_colors = SpectrumColors()

        fig = plt.figure()
        graph, = self._plot_vs_lnr_single(0, product_name, y_scale, plt_colors)

        stamp_steps = np.linspace(0, len(self.settings.steps)-1, num_fixed)
        for i in stamp_steps:
            new_graph, = self._plot_vs_lnr_single(
                int(i), product_name, y_scale, plt_colors)
            new_graph.set_alpha(0.5)

        def animate(i):
            update_list = [self._plot_vs_lnr_single(
                i, product_name, y_scale, plt_colors, set_to=graph), plt.legend()]

            return update_list

        anim = animation.FuncAnimation(fig, animate,
                                       frames=len(self.settings.steps), interval=50)

        plt_name = self.get_plt_name(product_name.replace("/", "_"))

        plt.xscale("log", base=10)

        plt.title(plt_name)
        plt.xlabel("particle radius [µm]")
        plt.ylabel(y_label)

        output_nums = [i[product_name] for i in self._output]
        plt.xlim((np.min(self.settings.radius_bins_edges) * si.metres / si.micrometres,
                  np.max(self.settings.radius_bins_edges) * si.metres / si.micrometres))
        plt.ylim((0, np.max(output_nums) * y_scale))

        anim.save(f"{plt_name}.gif", fps=30)

        if show:
            try:
                assert get_ipython().__class__.__name__ == "ZMQInteractiveShell"
                display(HTML(anim.to_html5_video()))
                plt.close()
            except:
                plt.show()


def generate_dm_dlnr_animation(evaluate_relaxed_velocity=True, tau=100*si.seconds):
    settings = Settings(
        n_sd=2**18, max_t=7200, evaluate_relaxed_velocity=evaluate_relaxed_velocity, tau=tau)
    simulation = Simulation(settings)

    simulation.run()

    simulation.plot_vs_lnr_animation(
        "dm/dlnr", si.kilograms/si.grams, "dm/dlnr [g/m^3/(unit dr/r)]", show=False, num_fixed=8)


if __name__ == "__main__":
    generate_dm_dlnr_animation(evaluate_relaxed_velocity=False)
    for tau in [1, 10, 100]:
        generate_dm_dlnr_animation(
            evaluate_relaxed_velocity=True, tau=tau)

# simulation.plot_vs_lnr_animation("fall_vel", 1, "fall velocity [m/s]")
