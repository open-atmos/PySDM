from typing import Any, Type, Union
from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumColors
import matplotlib.pyplot as plt
from PySDM.backends import CPU, GPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import ParticleVolumeVersusRadiusLogarithmSpectrum, RadiusBinnedNumberAveragedTerminalVelocity, WallTime
from PySDM.physics import si
from open_atmos_jupyter_utils import show_plot
from PySDM_examples.Bhalla_2023.settings_0D import Settings


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

        products = (
            ParticleVolumeVersusRadiusLogarithmSpectrum(
                self.settings.radius_bins_edges, name="dv/dlnr"
            ),
            RadiusBinnedNumberAveragedTerminalVelocity(
                self.settings.radius_bins_edges, name="terminal_vel"
            ),
            WallTime(),
        )

        self.particulator = self.builder.build(attributes, products)

        # if hasattr(self.settings, "u_term") and "terminal velocity" in self.particulator.attributes:
        #     self.particulator.attributes["terminal velocity"].approximation = self.settings.u_term(
        #         self.particulator
        #     )

        # for observer in observers:
        #     self.particulator.observers.append(observer)

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

            self._output.append(output_vals)

        self._exec_time = self.particulator.products["wall time"].get()
        self.done = True

    def get_plt_name(self, plot_var: str)->str:
        return f"0D {plot_var} v_t-{self.builder.get_attribute('terminal velocity').approximation.__class__.__name__} n_sd-{self.settings.n_sd} kernel-{self.settings.kernel.__class__.__name__}"

    # TODO: make this a more generic function which can plot anything with domain ln r
    def plot_vs_lnr(self, product_name: str, y_scale: float = 1, y_label: Union[str, None] = None):
        assert self.done

        y_label = y_label or product_name

        plt_colors = SpectrumColors()

        for i in range(len(self.settings.steps)):
            plt.step(
                self.settings.radius_bins_edges[:-
                                                1] * si.metres / si.micrometres,
                self._output[i][product_name] * y_scale,
                where="post",
                label=f"t = {self.settings.steps[i]}s",
                color=plt_colors(
                    self.settings.steps[i] /
                    (self.settings.output_steps[-1] * self.settings.dt)
                ),
            )

        plt_name = self.get_plt_name(product_name.replace("/", "_"))

        plt.xscale("log", base=10)

        plt.title(plt_name)
        plt.xlabel("particle radius [µm]")
        plt.ylabel(y_label)
        plt.legend()

        show_plot(filename=f"{plt_name}.pdf")


if __name__ == "__main__":
    settings = Settings(n_sd=2**18)
    simulation = Simulation(settings)

    simulation.run()

    simulation.plot_vs_lnr("dm/dlnr", si.kilograms/si.grams, "dm/dlnr [g/m^3/(unit dr/r)]")

    simulation.plot_vs_lnr("terminal_vel", 1, "terminal velocity [m/s]")
