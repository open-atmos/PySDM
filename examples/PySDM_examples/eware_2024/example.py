import os
from typing import Optional

import numpy as np

from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import ParticleVolumeVersusRadiusLogarithmSpectrum, WallTime, CollisionRateDeficitPerGridbox

from pystrict import strict

from PySDM import Formulae
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.initialisation import spectra
from PySDM.physics import si


import matplotlib
from matplotlib import pyplot
from open_atmos_jupyter_utils import show_plot
from packaging import version
from PySDM_examples.Shima_et_al_2009.error_measure import error_measure

from PySDM.physics.constants import si

_matplotlib_version_3_3_3 = version.parse("3.3.0")
_matplotlib_version_actual = version.parse(matplotlib.__version__)





# @strict
class Settings:
    def __init__(self:int,steps: Optional[list] = None):
        steps = steps or [0, 1200, 2400, 3600]
        self.formulae = Formulae()
        self.n_sd = 2**13
        self.n_part = 2**23 / si.metre**3
        self.X0 = self.formulae.trivia.volume(radius=30.531 * si.micrometres)
        self.dv = 1e6 * si.metres**3
        self.norm_factor = self.n_part * self.dv
        self.rho = 1000 * si.kilogram / si.metre**3
        self.dt = 1 * si.seconds
        self.adaptive = False
        self.steps = steps
        self.kernel = Golovin(b=1.5e3 / si.second)
        self.spectrum = spectra.Exponential(norm_factor=self.norm_factor, scale=self.X0)
        self.radius_bins_edges = np.logspace(
            np.log10(10 * si.um), np.log10(5e3 * si.um), num=128, endpoint=True
        )

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self.steps]



class SpectrumColors:
    def __init__(self, begining="#2cbdfe", end="#b317b1"):
        self.b = begining
        self.e = end

    def __call__(self, value: float):
        bR, bG, bB = int(self.b[1:3], 16), int(self.b[3:5], 16), int(self.b[5:7], 16)
        eR, eG, eB = int(self.e[1:3], 16), int(self.e[3:5], 16), int(self.e[5:7], 16)
        R = bR + int((eR - bR) * value)
        G = bG + int((eG - bG) * value)
        B = bB + int((eB - bB) * value)
        result = f"#{hex(R)[2:4]}{hex(G)[2:4]}{hex(B)[2:4]}"
        return result


class SpectrumPlotter:
    def __init__(self, settings, title=None, grid=True, legend=True, log_base=10):
        self.settings = settings
        self.format = "pdf"
        self.colors = SpectrumColors()
        self.smooth = False
        self.smooth_scope = 2
        self.legend = legend
        self.grid = grid
        self.title = title
        self.xlabel = "particle radius [µm]"
        self.ylabel = "dm/dlnr [g/m^3/(unit dr/r)]"
        self.log_base = log_base
        self.ax = pyplot
        self.fig = pyplot
        self.finished = False

    def finish(self):
        if self.finished:
            return
        self.finished = True
        if self.grid:
            self.ax.grid()

        base_arg = {
            "base"
            + (
                "x" if _matplotlib_version_actual < _matplotlib_version_3_3_3 else ""
            ): self.log_base
        }
        if self.title is not None:
            try:
                self.ax.title(self.title)
            except TypeError:
                self.ax.set_title(self.title)
        try:
            self.ax.xscale("log", **base_arg)
            self.ax.xlabel(self.xlabel)
            self.ax.ylabel(self.ylabel)
        except AttributeError:
            self.ax.set_xscale("log", **base_arg)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
        if self.legend:
            self.ax.legend()

    def show(self):
        self.finish()
        pyplot.tight_layout()
        show_plot()

    def save(self, file):
        self.finish()
        pyplot.savefig(file, format=self.format)

    def plot(
        self, spectrum, t, label=None, color=None, title=None, add_error_to_label=False
    ):
        error = self.plot_analytic_solution(self.settings, t, spectrum, title)
        if label is not None and add_error_to_label:
            label += f" error={error:.4g}"
        self.plot_data(self.settings, t, spectrum, label, color)
        return error

    def plot_analytic_solution(self, settings, t, spectrum, title):
        if t == 0:
            analytic_solution = settings.spectrum.size_distribution
        else:

            def analytic_solution(x):
                return settings.norm_factor * settings.kernel.analytic_solution(
                    x=x, t=t, x_0=settings.X0, N_0=settings.n_part
                )

        volume_bins_edges = self.settings.formulae.trivia.volume(
            settings.radius_bins_edges
        )
        dm = np.diff(volume_bins_edges)
        dr = np.diff(settings.radius_bins_edges)

        pdf_m_x = volume_bins_edges[:-1] + dm / 2
        pdf_m_y = analytic_solution(pdf_m_x)

        pdf_r_x = settings.radius_bins_edges[:-1] + dr / 2
        pdf_r_y = pdf_m_y * dm / dr * pdf_r_x

        x = pdf_r_x * si.metres / si.micrometres
        y_true = (
            pdf_r_y
            * self.settings.formulae.trivia.volume(radius=pdf_r_x)
            * settings.rho
            / settings.dv
            * si.kilograms
            / si.grams
        )

        self.ax.plot(x, y_true, color="black")

        if spectrum is not None:
            y = spectrum * si.kilograms / si.grams
            error = error_measure(y, y_true, x)
            self.title = (
                title or f"error measure: {error:.2f}"
            )  # TODO #327 relative error
            return error
        return None

    def plot_data(self, settings, t, spectrum, label, color):
        if self.smooth:
            scope = self.smooth_scope
            if t != 0:
                new = np.copy(spectrum)
                for _ in range(2):
                    for i in range(scope, len(spectrum) - scope):
                        new[i] = np.mean(spectrum[i - scope : i + scope + 1])
                    scope = 1
                    for i in range(scope, len(spectrum) - scope):
                        spectrum[i] = np.mean(new[i - scope : i + scope + 1])

            x = settings.radius_bins_edges[:-scope]
            dx = np.diff(x)
            self.ax.plot(
                (x[:-1] + dx / 2) * si.metres / si.micrometres,
                spectrum[:-scope] * si.kilograms / si.grams,
                label=label or f"t = {t}s",
                color=color
                or self.colors(t / (self.settings.output_steps[-1] * self.settings.dt)),
            )
        else:
            self.ax.step(
                settings.radius_bins_edges[:-1] * si.metres / si.micrometres,
                spectrum * si.kilograms / si.grams,
                where="post",
                label=label or f"t = {t}s",
                color=color
                or self.colors(t / (self.settings.output_steps[-1] * self.settings.dt)),
            )


def run(settings, backend, observers=()):
    env = Box(dv=settings.dv, dt=settings.dt)
    builder = Builder(
        n_sd=settings.n_sd, backend=backend, environment=env
    )
    builder.particulator.environment["rhod"] = 1.0
    attributes = {}
    sampling = ConstantMultiplicity(settings.spectrum)
    attributes["volume"], attributes["multiplicity"] = sampling.sample(settings.n_sd)
    coalescence = Coalescence(
        collision_kernel=settings.kernel, adaptive=settings.adaptive
    )
    builder.add_dynamic(coalescence)
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            settings.radius_bins_edges, name="dv/dlnr"
        ),
        WallTime(),
        CollisionRateDeficitPerGridbox(name='deficit'),
    )
    particulator = builder.build(attributes, products)

    for observer in observers:
        particulator.observers.append(observer)

    vals = {}
    deficit = 0
    particulator.products["wall time"].reset()
    for step in settings.output_steps:
        particulator.run(step - particulator.n_steps)
        vals[step] = particulator.products["dv/dlnr"].get()[0]
        vals[step][:] *= settings.rho
        deficit += particulator.products["deficit"].get()
        # deficit[step][:] *= settings.rho / settings.dv

    exec_time = particulator.products["wall time"].get()
    return vals, exec_time, deficit


def main(plot: bool, save: Optional[str]):
    with np.errstate(all="raise"):
        settings = Settings()

        settings.n_sd = 2**15

        states, _ = run(settings)

    with np.errstate(invalid="ignore"):
        plotter = SpectrumPlotter(settings)
        plotter.smooth = True
        for step, vals in states.items():
            _ = plotter.plot(vals, step * settings.dt)
            # assert _ < 200  # TODO #327
        if save is not None:
            n_sd = settings.n_sd
            plotter.save(save + "/" + f"{n_sd}_shima_fig_2" + "." + plotter.format)
        if plot:
            plotter.show()


if __name__ == "__main__":
    main(plot="CI" not in os.environ, save=None)



